"""Simulator of a simplified thermodynamic system for flexible geometries.

This simulator uses finite differences method (FDM) to approximate the
temperature changes in each Control Volume (CV) in a building.

The `execute_step_sim` method is responsible for calculating the supply air
temperature from the air handler(s) for the current simulation time step.

First, it calculates the recirculation air temperature for each air handler
unit (AHU). This is the area-weighted average temperature of all zones serviced
by that AHU.

It iterates through each Variable Air Volume (VAV) unit to find the average
temperature of the zones it serves (v_sensing_temp), weighted by the area of
each zone. It then calculates the recirculation temperature for each AHU by
taking an area-weighted average of the v_sensing_temp of all VAVs connected
to it. Finally, it uses this per-AHU recirculation temperature and the outside
ambient temperature to calculate the temperature of the air that will be
supplied by each AHU. This logic handles both single and multiple AHU
configurations.
"""

import collections
from collections.abc import Mapping, Sequence
from typing import NamedTuple

import gin
import numpy as np
import pandas as pd
import scipy.sparse
from absl import logging

from smart_buildings.smart_control.models.base_occupancy import BaseOccupancy
from smart_buildings.smart_control.proto import (
  smart_control_building_pb2,
  smart_control_reward_pb2,
)
from smart_buildings.smart_control.simulator import air_handler as air_handler_py
from smart_buildings.smart_control.simulator import building as building_py
from smart_buildings.smart_control.simulator import (
  constants,
  hvac_floorplan_based,
  simulator,
)
from smart_buildings.smart_control.simulator import (
  weather_controller as weather_controller_py,
)
from smart_buildings.smart_control.utils import (
  building_renderer,
  conversion_utils,
  visual_logger,
)

RewardInfo = smart_control_reward_pb2.RewardInfo

CVCoordinates = tuple[int, int]
ZoneId = tuple[int, int]


class _VavZones(NamedTuple):
  """The zones a VAV serves, rolled up once and reused later in the step."""

  assigned_zones: list[str]
  # Total floor area of the assigned zones, in control volumes.
  area: float
  # Area-weighted average temperature across the assigned zones.
  sensing_temp: float


@gin.configurable
class SimulatorFlexibleGeometries(simulator.Simulator[str]):
  """Simulates thermodynamics of a building with flexible geometries.

  NOTE: post-refector

  Adds the floor-plan building family to Simulator: zones are identified by
  room name rather than by grid coordinate. The thermal step itself is the
  base class's, and is the same for every building family.
  """

  building: building_py.FloorPlanBasedBuilding
  _hvac: hvac_floorplan_based.FloorPlanBasedHvac

  def __init__(
      self,
      building: building_py.FloorPlanBasedBuilding,
      hvac: hvac_floorplan_based.FloorPlanBasedHvac,
      weather_controller: weather_controller_py.WeatherController,
      time_step_sec: float,
      start_timestamp: pd.Timestamp,
      capture_frames: bool = False,
      video_export_day: int | None = None,
      internal_gain_w_per_m2: float = 0.0,
      internal_gain_standby_fraction: float = 0.2,
      occupant_sensible_heat_w: float = 70.0,
  ):
    """Simulator init.

    Args:
      building: Refactored flexible FloorPlanBasedBuilding object controlling
        the control volumes.
      hvac: Hvac for the building.
      weather_controller: Weather Controller for modelling ambient temperature.
      time_step_sec: Size of each time step in seconds.
      start_timestamp: Pandas timestamp representing start time for simulation.
      capture_frames: Retain one temperature frame per step for video export.
        Off by default: each frame is the whole grid, so a long episode
        would accumulate gigabytes.
      video_export_day: Simulated day on which to write a video, or None
        for no export. Export needs the optional mediapy package.
      internal_gain_w_per_m2: Heat from lighting, equipment and people, in
        W per m2 of zone floor area. Zero reproduces the original behaviour,
        which modelled no internal gains at all.
      internal_gain_standby_fraction: Fraction of the gain that persists
        outside comfort hours, covering standby equipment and security
        lighting.
      occupant_sensible_heat_w: Sensible heat in W given off by one person,
        applied to the zone that person is actually in. A seated adult puts
        out roughly 70 W sensible. Set to zero to disable occupant gains, so
        that thermal load ignores the occupancy model.
    """
    self.building = building
    self._hvac = hvac
    self._occupant_sensible_heat_w = occupant_sensible_heat_w
    # Set by SimulatorBuilding, which owns the occupancy model. Left None if
    # nothing wires it up, in which case occupant gains are simply skipped.
    self._occupancy = None

    zone_identifiers = [
        z
        for z in self.building.room_dict.keys()
        if not constants.is_non_physical_space(z)
    ]
    self._hvac.initialize_zone_identifier(
        zone_identifiers, self.building.custom_zone_to_vavs
    )

    super().__init__(
        self.building,
        self._hvac,
        weather_controller,
        time_step_sec,
        start_timestamp,
    )

    render_zones = np.copy(self.building.floor_plan)
    render_zones[render_zones == 2] = 0

    renderer = building_renderer.BuildingRenderer(render_zones, 1)

    self._log_and_plotter = visual_logger.VisualLogger(renderer)
    # Diagnostics, off by default. See execute_step_sim.
    self._capture_frames = capture_frames
    self._video_export_day = video_export_day
    self._internal_gain_w_per_m2 = internal_gain_w_per_m2
    self._internal_gain_standby_fraction = internal_gain_standby_fraction
    self._couple_air_side()
    self.reset()

  def _couple_air_side(self) -> None:
    """Hands the VAVs' air streams to the solver to be applied implicitly.

    A VAV's reheat coil delivers a duty the water loop fixes, but its air
    stream delivers `flow * rho * cp * (T_supply - Tbar)`, which depends on the
    zone temperature it is trying to change. Given to the solver as watts it
    has to be evaluated before the step, and the zone is then thrown past the
    supply temperature and back whenever the air can turn the zone's heat
    capacity over faster than twice a timestep -- which on these floor plans it
    can, several times over. So the two are separated: the coil stays in
    input_q, and the air stream is described to the solver here, once, as where
    each VAV's power lands and which temperature it reads.

    The description is the same one apply_thermal_power_zone would act on: a
    VAV splits its output between its zones by floor area and each zone spreads
    its share over its diffusers, and it reads back the mean over every CV it
    serves, which is that same area weighting applied to per-zone means.
    """
    room_dict = self.building.room_dict
    zones_with_temps = self.building.get_zone_average_temps()
    n_columns = self.building.temp.shape[1]

    # A VAV applies power to and reads temperature from exactly the CVs of the
    # zones it serves, so both matrices share one sparsity pattern and differ
    # only in their values.
    self._coupled_vavs = []
    empty = np.empty(0, dtype=int)
    cvs, vav_columns, applied, sensed = [empty], [empty], [empty], [empty]
    for v_id in self._hvac.vavs:
      zones = [
          z for z in self._hvac.get_zones_for_vav(v_id) if z in zones_with_temps
      ]
      if not zones:
        continue
      column = len(self._coupled_vavs)
      self._coupled_vavs.append(v_id)
      vav_area = sum(len(room_dict[z]) for z in zones)
      for zone in zones:
        rows, columns = np.transpose(room_dict[zone])
        cvs.append(rows * n_columns + columns)
        vav_columns.append(np.full(len(rows), column))
        applied.append(
            self.building.diffusers[rows, columns] * (len(rows) / vav_area)
        )
        sensed.append(np.full(len(rows), 1.0 / vav_area))

    shape = (self.building.temp.size, len(self._coupled_vavs))
    position = (np.concatenate(cvs), np.concatenate(vav_columns))
    # Most CVs in a zone carry no diffuser, so most of applied is structurally
    # zero and only the pattern of sensed is really this dense.
    applied = scipy.sparse.csc_matrix(
        (np.concatenate(applied), position), shape=shape
    )
    applied.eliminate_zeros()
    self._air_side_sensed = scipy.sparse.csc_matrix(
        (np.concatenate(sensed), position), shape=shape
    )
    self._solver.couple_air_side(applied, self._air_side_sensed)

  def reset(self):
    """Resets the simulation to its initial configuration."""
    super().reset()

    # Clear the visual logger so per-step thermal-grid copies do not accumulate
    # across episodes for the lifetime of a cached environment (OOM source).
    # The base Simulator constructor calls self.reset() before _log_and_plotter
    # is assigned in __init__, so guard against the attribute not existing yet.
    if getattr(self, "_log_and_plotter", None) is not None:
      self._log_and_plotter.reset()

    # VAV zone-air-temperature sensors are initialized to 0 K and are only
    # populated from the building's thermal field inside execute_step_sim().
    # Without priming them here, the very first observation (from env.reset())
    # reports 0 K zone temperatures, which normalize to a -59 sentinel. Mirror
    # the per-VAV area-weighted sensing logic from execute_step_sim() (the grid
    # is uniform at reset so weighting is not strictly required, but it is kept
    # for generality), and set the sensor value directly without running the
    # thermostat so we avoid mutating control/hysteresis state before step 0.
    avg_temps = self.building.get_zone_average_temps()
    room_dict = self.building.room_dict
    room_areas = {z: len(coords) for z, coords in room_dict.items()}
    for v_id, vav in self._hvac.vavs.items():
      assigned_zones = [
          z for z in self._hvac.get_zones_for_vav(v_id) if z in avg_temps
      ]
      if not assigned_zones:
        continue
      v_area = sum(room_areas[z] for z in assigned_zones)
      v_sensing_temp = (
          sum(avg_temps[z] * room_areas[z] for z in assigned_zones) / v_area
      )
      vav.zone_air_temperature = v_sensing_temp

  @property
  def time_step_sec(self) -> float:
    return self._time_step_sec


  @property
  def hvac(self) -> hvac_floorplan_based.FloorPlanBasedHvac:
    return self._hvac

  @property
  def current_timestamp(self) -> pd.Timestamp:
    return self._current_timestamp

  def setup_step_sim(self) -> None:
    """Disables the base Simulator setup step.

    This method is overridden with a `pass` because SimulatorFlexibleGeometries
    handles VAV updates more robustly in `execute_step_sim` using weighted
    averages for shared zones and custom VAV IDs. The base class's
    `setup_step_sim` assumes a 1:1 mapping between room names and VAV IDs,
    which causes a KeyError when custom mappings are used.
    """

  def step_sim(self, video_filename: str = "sample.mp4") -> None:
    """Steps simulation, executing one time step."""
    self.execute_step_sim(video_filename)

  def execute_step_sim(
      self, video_filename: str = "sample.mp4"
  ) -> None:
    """This method should not change any actions set on smart devices."""

    current_ts = self._current_timestamp
    hvac = self._hvac
    room_dict = self.building.room_dict
    room_areas = {z: len(coords) for z, coords in room_dict.items()}
    avg_temps = self.building.get_zone_average_temps()
    ambient_temperature = self._weather_controller.get_current_temp(current_ts)

    ahu_temp_weighted_sum = collections.defaultdict(float)
    ahu_area_total = collections.defaultdict(float)
    vav_cached_data: dict[str, _VavZones] = {}
    vav_sensing_temp_map = {}

    for v_id, vav in hvac.vavs.items():
      ahu_id = vav.air_handler.device_id()
      assigned_zones = [
          z for z in hvac.get_zones_for_vav(v_id) if z in avg_temps
      ]
      if not assigned_zones:
        continue

      v_area = sum(room_areas[z] for z in assigned_zones)
      v_sensing_temp = (
          sum(avg_temps[z] * room_areas[z] for z in assigned_zones) / v_area
      )
      vav.update_settings(v_sensing_temp, current_ts)

      vav_cached_data[v_id] = _VavZones(
          assigned_zones=assigned_zones,
          area=v_area,
          sensing_temp=v_sensing_temp,
      )
      vav_sensing_temp_map[v_id] = v_sensing_temp

      ahu_temp_weighted_sum[ahu_id] += v_sensing_temp * v_area
      ahu_area_total[ahu_id] += v_area

    if isinstance(hvac.air_handler, air_handler_py.AirHandlerSystem):
      ahus = hvac.air_handler.ahus
    else:
      ahus = [hvac.air_handler]

    recirculation_temps = {}
    for ahu in ahus:
      ahu_id = ahu.device_id()
      if ahu_area_total.get(ahu_id, 0) > 0:
        recirculation_temps[ahu_id] = (
            ahu_temp_weighted_sum[ahu_id] / ahu_area_total[ahu_id]
        )
      else:
        recirculation_temps[ahu_id] = self.building.temp.mean()

    if isinstance(hvac.air_handler, air_handler_py.AirHandlerSystem):
      supply_air_temp = hvac.air_handler.get_supply_air_temp(
          recirculation_temps, ambient_temperature
      )
    else:
      # Single AHU case
      supply_air_temp = hvac.air_handler.get_supply_air_temp(
          recirculation_temps[hvac.air_handler.device_id()], ambient_temperature
      )

    convection_coefficient = (
        self._weather_controller.get_air_convection_coefficient(current_ts)
    )

    # Reset the air handler and hws flow rate demand before accumulating.
    hvac.air_handler.reset_demand()
    hvac.hot_water_system.reset_demand()

    # sum up all the VAV hot waterdemands for the current timestep
    # this needs to be calculated first before the output function is called,
    # since the flow rate of the entire system can only be determined if we know
    # the total demand from all VAVs
    for vav in hvac.vavs.values():
      hvac.hot_water_system.add_demand(vav.reheat_flow_factor)

    # The same argument applies to air. An air handler caps its total flow, so
    # the share each VAV actually receives can only be known once every VAV has
    # asked. Accumulating this in the output loop below is too late: the first
    # VAV would be evaluated against a handler holding no demand yet, and the cap
    # would only ever reach the fan bill, never the heat delivered. Zones would
    # be heated for the uncapped flow while the fan was billed for the capped
    # one, which creates energy from nothing in the agent's favour.
    air_flow_demand = collections.defaultdict(float)
    for vav in hvac.vavs.values():
      air_flow_demand[vav.air_handler.device_id()] += (
          vav.unrestricted_flow_rate_demand
      )
    for vav in hvac.vavs.values():
      demanded = air_flow_demand[vav.air_handler.device_id()]
      if demanded > 0:
        vav.supply_air_flow_ratio = min(
            1.0, vav.air_handler.max_air_flow_rate / demanded
        )
      else:
        vav.supply_air_flow_ratio = 1.0

    # Stores the total thermal energy to be applied to each physical zone
    zone_q_aggregate = collections.defaultdict(float)

    # The water loop's side of the reheat energy balance, accumulated here so
    # the boiler can be charged for exactly the heat the coils delivered. See
    # _calculate_return_water_temperature_from_reheat.
    total_reheat_w = 0.0
    coil_entering_air_temps = []

    # The air stream's side does not go into zone_q_aggregate at all. Its heat
    # depends on the zone temperature it is about to change, so it is handed to
    # the solver as a conductance and a supply temperature and settled inside
    # the timestep; see _couple_air_side. Only the coil, whose duty the water
    # loop fixes, is a known number of watts before the step.
    air_side_conductance = np.zeros(len(self._coupled_vavs))
    air_side_supply_temp = np.zeros(len(self._coupled_vavs))

    for column, v_id in enumerate(self._coupled_vavs):
      if v_id not in vav_cached_data:
        continue
      vav = hvac.vavs[v_id]
      cached_data = vav_cached_data[v_id]
      assigned_zones = cached_data.assigned_zones
      total_vav_area = cached_data.area

      # 2) thermal computation
      if isinstance(supply_air_temp, dict):
        current_supply_temp = supply_air_temp[vav.air_handler.device_id()]
      else:
        current_supply_temp = supply_air_temp
      air_side_supply_temp[column] = current_supply_temp

      reheat_w = vav.compute_reheat_energy_rate(
          current_supply_temp,
          hvac.hot_water_system.supply_water_temperature_sensor,
      )
      total_reheat_w += reheat_w
      coil_entering_air_temps.append(current_supply_temp)

      # A VAV with no damper or no fan reaches its zones with nothing at all,
      # neither air nor reheat, which is what compute_energy_applied_to_zone
      # says too. The boiler is still charged for the coil above, as it was.
      if vav.damper_setting > 0 and vav.max_air_flow_rate > 0:
        air_side_conductance[column] = (
            vav.flow_rate_demand
            * constants.AIR_DENSITY
            * constants.AIR_HEAT_CAPACITY
        )
        # 3) virtual vav distribution to zones proportional to area
        for z in assigned_zones:
          zone_q_aggregate[z] += reheat_w * room_areas[z] / total_vav_area

      # Update the air handler airflow demand by summing from all VAVs.
      if vav.flow_rate_demand > 0:
        vav.air_handler.add_demand(vav.flow_rate_demand)

    # Internal heat gains from lighting, equipment and people. A real office
    # runs at 15 to 25 W/m2 from these sources, and the model had none at all,
    # which is a large part of why the building needs so much heat in winter.
    #
    # They are added into zone_q_aggregate rather than applied separately
    # because apply_thermal_power_zone ASSIGNS input_q rather than accumulating
    # into it, so a second call would overwrite the HVAC term instead of adding
    # to it.
    if self._internal_gain_w_per_m2 > 0.0:
      fraction = (
          1.0
          if hvac.is_comfort_mode(current_ts)
          else self._internal_gain_standby_fraction
      )
      cv_area_m2 = (self.building.cv_size_cm / 100.0) ** 2
      watts_per_cv = self._internal_gain_w_per_m2 * cv_area_m2 * fraction
      for zone_name, coords in room_dict.items():
        if zone_name in ("exterior_space", "interior_wall"):
          continue
        zone_q_aggregate[zone_name] += watts_per_cv * len(coords)

    # People, applied to the zone each person is actually in.
    #
    # This is what ties the thermal load to the same per-zone stochastic
    # occupancy the reward is computed from. The internal gain above is
    # spatially uniform and a square wave in time, driven by a building-wide
    # comfort-mode flag, so on its own it carries none of the per-zone
    # variation the agent is graded against.
    #
    # average_zone_occupancy caches on (start, end), and this asks for the same
    # interval the reward path does, so the occupants' Markov chains are not
    # advanced a second time here. Advancing them twice would change every
    # trajectory without raising anything.
    if self._occupancy is not None and self._occupant_sensible_heat_w > 0.0:
      interval_end = current_ts + pd.Timedelta(self._time_step_sec, unit="s")
      for zone_name in room_dict:
        if zone_name in ("exterior_space", "interior_wall"):
          continue
        occupants = self._occupancy.average_zone_occupancy(
            zone_name, current_ts, interval_end
        )
        zone_q_aggregate[zone_name] += (
            occupants * self._occupant_sensible_heat_w
        )

      # apply aggregated power to building
    for z, q_total in zone_q_aggregate.items():
      self.building.apply_thermal_power_zone(z, q_total)

    # Update each control volume.
    self.finite_differences_timestep(
        ambient_temperature=ambient_temperature,
        convection_coefficient=convection_coefficient,
        air_side_conductance=air_side_conductance,
        supply_temperature=air_side_supply_temp,
    )

    # Now that the field has settled, what the air stream actually delivered is
    # the same expression the VAV would have evaluated before the step, read at
    # the end of it instead. It is folded back into input_q so that stays what
    # its docstring says -- the power applied to each CV this step -- for the
    # renderer and for get_zone_thermal_energy_rate.
    air_side_q = air_side_conductance * (
        air_side_supply_temp
        - self._air_side_sensed.T @ self.building.temp.reshape(-1)
    )
    for column, v_id in enumerate(self._coupled_vavs):
      cached_data = vav_cached_data.get(v_id)
      if cached_data is None:
        continue
      for z in cached_data.assigned_zones:
        zone_q_aggregate[z] += (
            air_side_q[column] * room_areas[z] / cached_data.area
        )
    for z, q_total in zone_q_aggregate.items():
      self.building.apply_thermal_power_zone(z, q_total)

    # Simulate airflow
    self.building.apply_convection()

    hvac.hot_water_system.return_water_temperature_sensor = (
        self._calculate_return_water_temperature_from_reheat(
            total_reheat_w, coil_entering_air_temps
        )
    )
    hvac.hot_water_system.apply_heat_draw(
        ambient_temperature, self._time_step_sec
    )

    # A guard against solver divergence, not a physical bound.
    #
    # The bounds have to sit outside the range real weather reaches. Exterior
    # CVs are pinned to ambient rather than solved (the sweep assigns them
    # t_temp_inf) and they live in this same array, so a bound anywhere inside
    # the weather's range would not clip a divergence, it would rewrite the
    # boundary condition and warm the outdoor air. The coldest station here
    # spends a large part of its winter below 250 K, so 150 K is the floor.
    #
    # Nothing either the weather or a habitable building can reach comes near
    # these, so engaging them means the solve has diverged and the step is
    # already wrong. Say so rather than hiding it.
    if self.building.temp.min() < 150 or self.building.temp.max() > 500:
      logging.warning(
          "Temperature field left physical bounds at %s: min %.1f K, max"
          " %.1f K. The finite-difference solve has diverged.",
          self._current_timestamp,
          self.building.temp.min(),
          self.building.temp.max(),
      )
      self.building.temp = np.clip(self.building.temp, 150, 500)

    # Increment the timestamp.
    self._current_timestamp += pd.Timedelta(self._time_step_sec, unit="s")
    # Frame capture and video export are diagnostics, and are off unless a
    # caller asks for them. Left on they make a full episode impossible:
    #   * log() appends the whole grid every step, about 1.8 MB a frame, so
    #     a month-long episode accumulates well over 10 GB.
    #   * the export fired unconditionally at simulated day 4 and imports
    #     mediapy, which is not a dependency, so every episode died there
    #     with ModuleNotFoundError at step 1151.
    # A building simulation must not need a video library to reach the end
    # of its own episode.
    if self._capture_frames:
      self._log_and_plotter.log(self.building.temp)

    if (
        self._video_export_day is not None
        and self.current_timestamp
        == self._start_timestamp + pd.Timedelta(days=self._video_export_day)
    ):
      self.get_video(path=constants.VIDEO_PATH_ROOT + video_filename)

  def _calculate_return_water_temperature_from_reheat(
      self, total_reheat_w: float, coil_entering_air_temps: Sequence[float]
  ) -> float:
    """Returns the return water temperature implied by the heat the coils gave.

    Deliberately not an override of Simulator._calculate_return_water_temperature:
    that one is keyed by zone, and here a zone can be served by several VAVs.

    The boiler bills c_w * rho_w * flow * (T_supply - T_return), so this
    temperature is what decides the gas charge. It must therefore come from an
    energy balance on the water loop and nothing else:

        T_return = T_supply - Q_reheat / (c_w * rho_w * flow)

    The previous version instead averaged the VAVs' *supply air* temperatures,
    weighted by reheat flow factor. Those two quantities coincide only in the
    deeply water-limited regime. Everywhere else the boiler was charged for a
    temperature drop unrelated to the heat it had delivered -- under-billed by
    3.5x at full airflow. As airflow falls, Vav._reheat_effectiveness saturates
    at its cap of 1, the air leaves at the water temperature, the inferred
    return water temperature converges on the supply water temperature, and the
    gas charge collapses to zero while the coils keep putting hundreds of kW
    into the zones. Holding static pressure near 100 Pa reached the same zone
    temperature as full flow for 1/700th of the gas, which is reachable from the
    action space as a single constant.

    Args:
      total_reheat_w: Air-side heat delivered by every reheat coil this step, in
        W. Negative when the water is cooler than the air entering the coils.
      coil_entering_air_temps: Air temperature in K entering each active coil.

    Returns:
      The calculated return water temperature.
    """
    hws = self._hvac.hot_water_system
    water_capacity_rate = (
        constants.WATER_HEAT_CAPACITY
        * constants.WATER_DENSITY
        * hws.total_flow_rate
    )
    if water_capacity_rate <= 0.0 or not coil_entering_air_temps:
      # With no flow the loop is not exchanging heat, so the sensor holds.
      return hws.return_water_temperature_sensor

    supply_temp = hws.supply_water_temperature_sensor
    return_temp = supply_temp - total_reheat_w / water_capacity_rate

    # The water leaves a coil somewhere between its own inlet temperature and
    # the air's: it approaches the air temperature but cannot cross it, in
    # either direction. Bounding against the most extreme air temperature keeps
    # this true for the mixed return of coils running at different conditions.
    air_temp = (
        min(coil_entering_air_temps)
        if total_reheat_w >= 0.0
        else max(coil_entering_air_temps)
    )
    low, high = min(supply_temp, air_temp), max(supply_temp, air_temp)
    return min(high, max(low, return_temp))

  def _get_zone_reward_info(
      self,
      occupancy_function: BaseOccupancy,
      zone_coords: str,
      zone_id: str,
      zone_air_temperature: float,
  ) -> RewardInfo.ZoneRewardInfo:
    """Returns a message with zone data to compute the instantaneous reward.

    This assumes all VAVs for a zone share the same setpoint schedule,
    so the schedule is taken from the first VAV. If this assumption
    changes with local control, this method will need updating. Airflow from all
    VAVs feeding the zone is aggregated into a single virtual VAV to compute
    the correct airflow setpoint and actual airflow.

    Args:
      occupancy_function: The occupancy function to use for the zone.
      zone_coords: The coordinates of the zone.
      zone_id: The identifier of the zone.
      zone_air_temperature: The air temperature in the zone.

    Returns:
      A ZoneRewardInfo message.
    """
    room_dict = self.building.room_dict
    room_areas = {z: len(coords) for z, coords in room_dict.items()}
    vav_ids = self._hvac.get_vav_ids_for_zone(zone_coords)
    vavs = [self._hvac.vavs[v_id] for v_id in vav_ids]
    if not vavs:
      logging.warning("Zone %s has no servicing VAVs.", zone_id)
      return RewardInfo.ZoneRewardInfo()
    schedule = vavs[0].thermostat.get_setpoint_schedule()
    heating_setpoint_temperature, cooling_setpoint_temperature = (
        schedule.get_temperature_window(self._current_timestamp)
    )
    # Aggregate Capacity and Demand with Area Weighting
    zone_total_flow_setpoint = 0.0
    zone_total_flow_actual = 0.0
    for v in vavs:
      # How much of this specific VAV's total footprint belongs to this room?
      serviced_zones = self._hvac.get_zones_for_vav(v.device_id())
      total_vav_area = sum(room_areas[z] for z in serviced_zones)
      area_share = room_areas[zone_coords] / total_vav_area
      zone_total_flow_setpoint += v.max_air_flow_rate * area_share
      zone_total_flow_actual += v.flow_rate_demand * area_share

    average_occupancy = occupancy_function.average_zone_occupancy(
        zone_id,
        self._current_timestamp,
        self._current_timestamp + pd.Timedelta(self._time_step_sec, unit="s"),
    )
    return RewardInfo.ZoneRewardInfo(
        heating_setpoint_temperature=heating_setpoint_temperature,
        cooling_setpoint_temperature=cooling_setpoint_temperature,
        zone_air_temperature=zone_air_temperature,
        air_flow_rate_setpoint=zone_total_flow_setpoint,
        air_flow_rate=zone_total_flow_actual,
        average_occupancy=average_occupancy,
    )

  def _get_zone_reward_infos(
      self, occupancy_function: BaseOccupancy
  ) -> Mapping[str, RewardInfo.ZoneRewardInfo]:
    """Returns a map of messages with zone data.

    This data is used to compute the instantaneous reward.

    Args:
      occupancy_function: An occupancy function.
    """
    zone_reward_infos = {}
    for (
        zone_coords,
        zone_air_temperature,
    ) in self.building.get_zone_average_temps().items():
      zone_id = zone_coords
      zone_reward_infos[zone_id] = self._get_zone_reward_info(
          occupancy_function, zone_coords, zone_id, zone_air_temperature
      )
    return zone_reward_infos

  def _get_air_handler_reward_infos(
      self,
  ) -> Mapping[str, RewardInfo.AirHandlerRewardInfo]:
    """Returns a map of messages with air handler data.

    This data is used to compute the instantaneous reward.
    """
    air_handler_reward_infos = {}
    air_handler_id = self._hvac.air_handler.device_id()
    blower_electrical_energy_rate = (
        self._hvac.air_handler.compute_intake_fan_energy_rate()
        + self._hvac.air_handler.compute_exhaust_fan_energy_rate()
    )
    ambient_temp = self._weather_controller.get_current_temp(
        self._current_timestamp
    )
    avg_temps = self.building.get_zone_average_temps()
    room_dict = self.building.room_dict
    room_areas = {z: len(coords) for z, coords in room_dict.items()}

    ahu_temp_weighted_sum = collections.defaultdict(float)
    ahu_area_total = collections.defaultdict(float)

    for v_id, vav in self._hvac.vavs.items():
      ahu_id = vav.air_handler.device_id()
      assigned_zones = [
          z for z in self._hvac.get_zones_for_vav(v_id) if z in avg_temps
      ]
      if not assigned_zones:
        continue

      v_area = sum(room_areas[z] for z in assigned_zones)
      if v_area > 0:
        v_sensing_temp = (
            sum(avg_temps[z] * room_areas[z] for z in assigned_zones) / v_area
        )
        ahu_temp_weighted_sum[ahu_id] += v_sensing_temp * v_area
        ahu_area_total[ahu_id] += v_area

    if isinstance(self._hvac.air_handler, air_handler_py.AirHandlerSystem):
      ahus = self._hvac.air_handler.ahus
    else:
      ahus = [self._hvac.air_handler]

    recirculation_temps = {}
    for ahu in ahus:
      ahu_id = ahu.device_id()
      if ahu_area_total.get(ahu_id, 0) > 0:
        recirculation_temps[ahu_id] = (
            ahu_temp_weighted_sum[ahu_id] / ahu_area_total[ahu_id]
        )
      else:
        recirculation_temps[ahu_id] = self.building.temp.mean()

    if isinstance(self._hvac.air_handler, air_handler_py.AirHandlerSystem):
      recirculation_input = recirculation_temps
    else:
      # Single AHU case
      recirculation_input = recirculation_temps[
          self._hvac.air_handler.device_id()
      ]

    # The metered quantity, not the heat moved. A vapour compression cycle
    # moves several units of heat per unit of electricity, so the two differ by
    # the COP. compute_thermal_energy_rate stays the raw thermal load, because
    # that is what the VAVs account against; the division by COP belongs here,
    # at the point where the load is billed as electricity.
    air_conditioning_electrical_energy_rate = (
        self._hvac.air_handler.compute_cooling_electrical_energy_rate(
            recirculation_input, ambient_temp
        )
    )
    air_handler_reward_info = RewardInfo.AirHandlerRewardInfo(
        blower_electrical_energy_rate=blower_electrical_energy_rate,
        air_conditioning_electrical_energy_rate=air_conditioning_electrical_energy_rate,  # pylint: disable=line-too-long
    )
    air_handler_reward_infos[air_handler_id] = air_handler_reward_info
    return air_handler_reward_infos

  def _get_hws_reward_infos(
      self,
  ) -> tuple[
      Mapping[str, RewardInfo.BoilerRewardInfo | RewardInfo.HeatPumpRewardInfo],
      smart_control_building_pb2.DeviceInfo.DeviceType.ValueType,
  ]:
    """Returns a map of messages with hot water system data.

    This data is used to compute the instantaneous reward.
    """
    hws_reward_infos = {}
    hws_id = self._hvac.hot_water_system.device_id()
    return_water_temp = (
        self._hvac.hot_water_system.return_water_temperature_sensor
    )
    heating_energy_rate = (
        self._hvac.hot_water_system.compute_thermal_energy_rate(
            return_water_temp,
            self._weather_controller.get_current_temp(self._current_timestamp),
        )
    )
    pump_electrical_energy_rate = (
        self._hvac.hot_water_system.compute_pump_power()
    )
    hws_device_type = self._hvac.hot_water_system.heat_source_device_type
    device_type = smart_control_building_pb2.DeviceInfo.DeviceType
    if hws_device_type == device_type.BLR:
      hws_reward_info = RewardInfo.BoilerRewardInfo(
          natural_gas_heating_energy_rate=heating_energy_rate,
          pump_electrical_energy_rate=pump_electrical_energy_rate,
      )
    elif hws_device_type == device_type.ASHP:
      hws_reward_info = RewardInfo.HeatPumpRewardInfo(
          electricity_heating_energy_rate=heating_energy_rate,
          pump_electrical_energy_rate=pump_electrical_energy_rate,
      )
    else:
      raise ValueError(
          f"Unsupported heat source device type: {hws_device_type}"
      )

    hws_reward_infos[hws_id] = hws_reward_info
    return hws_reward_infos, hws_device_type

  def reward_info(self, occupancy_function: BaseOccupancy) -> RewardInfo:
    """Returns a message with data to compute the instantaneous reward."""
    start_time_stamp = self._current_timestamp
    end_time_stamp = start_time_stamp + pd.Timedelta(
        self._time_step_sec, unit="s"
    )

    # get zone data
    zone_reward_infos = self._get_zone_reward_infos(occupancy_function)

    # get air handler info
    air_handler_reward_infos = self._get_air_handler_reward_infos()

    # get hot water system info
    hws_reward_infos, hws_device_type = self._get_hws_reward_infos()

    reward_info_args = {
        "start_timestamp": conversion_utils.pandas_to_proto_timestamp(
            start_time_stamp
        ),
        "end_timestamp": conversion_utils.pandas_to_proto_timestamp(
            end_time_stamp
        ),
        "zone_reward_infos": zone_reward_infos,
        "air_handler_reward_infos": air_handler_reward_infos,
    }

    if hws_device_type == smart_control_building_pb2.DeviceInfo.DeviceType.BLR:
      reward_info_args["boiler_reward_infos"] = hws_reward_infos
    elif (
        hws_device_type == smart_control_building_pb2.DeviceInfo.DeviceType.ASHP
    ):
      reward_info_args["heat_pump_reward_infos"] = hws_reward_infos
    else:
      raise ValueError(
          f"Unsupported heat source device type: {hws_device_type}"
      )
    return RewardInfo(**reward_info_args)

  def get_video(self, path: str) -> None:
    """Wraps the get_video function from the visual_logger.

    Args:
      path: path to desired video directory (cns, etc.).

    Returns:
      None
    """

    self._log_and_plotter.get_video(
        file_path=path, fps=12, vmin=280, vmax=300, cmap="rainbow"
    )
