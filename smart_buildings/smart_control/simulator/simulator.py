"""Simulator of a simplified building and HVAC devices."""

from collections.abc import Mapping
from typing import cast

import gin
import numpy as np
import pandas as pd

from smart_buildings.smart_control.models.base_occupancy import BaseOccupancy
from smart_buildings.smart_control.proto import (
  smart_control_building_pb2,
  smart_control_reward_pb2,
)
from smart_buildings.smart_control.simulator import building as building_py
from smart_buildings.smart_control.simulator import (
  finite_difference_solver as fd_solver,
)
from smart_buildings.smart_control.simulator import hvac as hvac_py
from smart_buildings.smart_control.simulator import (
  weather_controller as weather_controller_py,
)
from smart_buildings.smart_control.utils import conversion_utils

RewardInfo = smart_control_reward_pb2.RewardInfo

CVCoordinates = tuple[int, int]
ZoneId = tuple[int, int]

@gin.configurable
class Simulator[ZoneKeyT]:
  """Simulates thermodynamics of a building.

  Each timestep is a backward-Euler finite difference step over the control
  volume grid, assembled and factorized once by FiniteDifferenceSolver.
  """

  # Left unbound in the zone key: this class only forwards zone identifiers
  # between the building and the HVAC, never interprets them. Subclasses
  # dedicated to one building family narrow these.
  building: building_py.BaseSimulatorBuilding[ZoneKeyT]
  _hvac: hvac_py.BaseHvac[ZoneKeyT]

  def __init__(
      self,
      building: building_py.BaseSimulatorBuilding[ZoneKeyT],
      hvac: hvac_py.BaseHvac[ZoneKeyT],
      weather_controller: weather_controller_py.WeatherController,
      time_step_sec: float,
      start_timestamp: pd.Timestamp,
  ):
    """Simulator init.

    Args:
      building: Building object controlling the control volumes.
      hvac: Hvac for the building.
      weather_controller: Weather Controller for modelling ambient temperature.
      time_step_sec: Size of each time step in seconds.
      start_timestamp: Pandas timestamp representing start time for simulation.
    """
    self.building = building
    self._hvac = hvac
    self._weather_controller = weather_controller
    self._time_step_sec = time_step_sec
    self._start_timestamp = start_timestamp
    # The matrix depends only on geometry, materials and the timestep, none of
    # which reset() disturbs, so one assembly lasts the whole episode. It is
    # built before the first reset because reset clears the solver's own state.
    self._solver = fd_solver.FiniteDifferenceSolver(
        self.building, time_step_sec
    )
    self.reset()

  def reset(self):
    """Resets the simulation to its initial configuration."""
    self.building.reset()
    self._hvac.reset()
    self._solver.reset(self.building.temp)
    self._current_timestamp = self._start_timestamp

  @property
  def hvac(self) -> hvac_py.BaseHvac[ZoneKeyT]:
    return self._hvac

  @property
  def weather_controller(self) -> weather_controller_py.WeatherController:
    return self._weather_controller

  @property
  def time_step_sec(self) -> float:
    return self._time_step_sec

  @property
  def start_timestamp(self) -> pd.Timestamp:
    return self._start_timestamp

  @property
  def current_timestamp(self) -> pd.Timestamp:
    return self._current_timestamp

  def finite_differences_timestep(
      self,
      *,
      ambient_temperature: float,
      convection_coefficient: float,
      air_side_conductance: np.ndarray | None = None,
      supply_temperature: np.ndarray | None = None,
  ) -> None:
    """Advances every CV's temperature to the end of the timestep.

    Args:
      ambient_temperature: Current temperature in K of external air.
      convection_coefficient: Current wind convection coefficient (W/m2/K).
      air_side_conductance: W/K of each air-side term a subclass registered
        with the solver, or None if it registered none.
      supply_temperature: Temperature in K each of those terms drives its zones
        towards.
    """
    self.building.temp = self._solver.solve(
        self.building.temp,
        self.building.input_q,
        ambient_temperature,
        convection_coefficient,
        air_side_conductance,
        supply_temperature,
    )

  def _calculate_return_water_temperature(
      self, zone_temps: Mapping[ZoneKeyT, float]
  ) -> float:
    numerator = 0.0
    denominator = 0.0
    for zone_id, vav in self._hvac.vavs.items():
      numerator += vav.reheat_valve_setting * zone_temps[zone_id]
      denominator += vav.reheat_valve_setting
    return numerator / (denominator + 1e-6)

  def setup_step_sim(self) -> None:
    """This method should not change the state of the building."""
    current_ts = self._current_timestamp
    hvac = self._hvac

    # Get the average temps in each zone. Assumes that the thermostat reads
    # the average room temperatures.
    avg_temps = self.building.get_zone_average_temps()

    for zone, zone_temp in avg_temps.items():
      vav = hvac.vavs[zone]

      # VAV update_setting handles the thermostat internally.
      vav.update_settings(zone_temp, current_ts)

  def execute_step_sim(self) -> None:
    """This method should not change any actions set on smart devices."""
    current_ts = self._current_timestamp
    hvac = self._hvac

    # Get the average temps in each zone. Assumes that the thermostat reads
    # the average room temperatures.
    avg_temps = self.building.get_zone_average_temps()

    # Recirculation temperature at the air handler is the global average.
    recirculation_temp = self.building.temp.mean()

    ambient_temperature = self._weather_controller.get_current_temp(current_ts)

    supply_air_temp = hvac.air_handler.get_supply_air_temp(
        recirculation_temp, ambient_temperature
    )

    convection_coefficient = (
        self._weather_controller.get_air_convection_coefficient(current_ts)
    )

    # Update each control volume.
    self.finite_differences_timestep(
        ambient_temperature=ambient_temperature,
        convection_coefficient=convection_coefficient,
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

    zone_supply_temp_map = {}
    # Heat the reheat coils deliver, for the tank's energy balance. The return
    # water temperature below is an average of discharge-air temperatures and
    # does not reproduce it. Only read with an energy-balanced tank: in the
    # legacy boiler every sensor read advances the tank temperature.
    energy_balance = hvac.hot_water_system.energy_balance
    coil_heat_w = 0.0

    # Iterate through each zone.
    for zone, zone_temp in avg_temps.items():
      vav = hvac.vavs[zone]

      if energy_balance:
        coil_heat_w += vav.compute_reheat_energy_rate(
            supply_air_temp,
            hvac.hot_water_system.supply_water_temperature_sensor,
        )
      q_zone, zone_supply_temp = vav.output(zone_temp, supply_air_temp)
      zone_supply_temp_map[zone] = zone_supply_temp

      # Update the air handler airflow demand by summing from all VAVs. The
      # demand goes to the VAV's own air handler rather than hvac.air_handler:
      # with an AirHandlerSystem the latter is the whole system, which has no
      # single airflow to charge the demand against.
      if vav.flow_rate_demand > 0:
        vav.air_handler.add_demand(vav.flow_rate_demand)

      # Apply the thermal energy to the zone.
      self.building.apply_thermal_power_zone(zone, q_zone)

    hvac.hot_water_system.return_water_temperature_sensor = (
        self._calculate_return_water_temperature(zone_supply_temp_map)
    )
    hvac.hot_water_system.apply_heat_draw(
        ambient_temperature, self._time_step_sec, coil_heat_w
    )

    # Increment the timestamp.
    self._current_timestamp += pd.Timedelta(self._time_step_sec, unit='s')

  def _get_zone_reward_info(
      self,
      occupancy_function: BaseOccupancy,
      zone_coords: ZoneKeyT,
      zone_id: str,
      zone_air_temperature: float,
  ) -> RewardInfo.ZoneRewardInfo:
    """Returns a messagde with zone data to compute the instantaneous reward."""
    schedule = self._hvac.vavs[zone_coords].thermostat.get_setpoint_schedule()
    heating_setpoint_temperature, cooling_setpoint_temperature = (
        schedule.get_temperature_window(self._current_timestamp)
    )
    air_flow_rate_setpoint = self._hvac.vavs[zone_coords].max_air_flow_rate
    air_flow_rate = self._hvac.air_handler.air_flow_rate
    average_occupancy = occupancy_function.average_zone_occupancy(
        zone_id,
        self._current_timestamp,
        self._current_timestamp + pd.Timedelta(self._time_step_sec, unit='s'),
    )
    zone_info = RewardInfo.ZoneRewardInfo(
        heating_setpoint_temperature=heating_setpoint_temperature,
        cooling_setpoint_temperature=cooling_setpoint_temperature,
        zone_air_temperature=zone_air_temperature,
        air_flow_rate_setpoint=air_flow_rate_setpoint,
        air_flow_rate=air_flow_rate,
        average_occupancy=average_occupancy,
    )
    return zone_info

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
      # This base implementation is only meaningful for the coordinate-keyed
      # Building; SimulatorFlexibleGeometries overrides it for room names.
      zone_id = conversion_utils.zone_coordinates_to_id(
          cast(CVCoordinates, zone_coords)
      )
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
    recirculation_temp = self.building.temp.mean()
    ambient_temp = self._weather_controller.get_current_temp(
        self._current_timestamp
    )
    air_conditioning_electrical_energy_rate = (
        self._hvac.air_handler.compute_thermal_energy_rate(
            recirculation_temp, ambient_temp
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

    if (
        self._hvac.hot_water_system.heat_source_device_type
        == smart_control_building_pb2.DeviceInfo.DeviceType.BLR
    ):
      hws_reward_info = RewardInfo.BoilerRewardInfo(
          natural_gas_heating_energy_rate=heating_energy_rate,
          pump_electrical_energy_rate=pump_electrical_energy_rate,
      )
    elif (
        self._hvac.hot_water_system.heat_source_device_type
        == smart_control_building_pb2.DeviceInfo.DeviceType.ASHP
    ):
      hws_reward_info = RewardInfo.HeatPumpRewardInfo(
          electricity_heating_energy_rate=heating_energy_rate,
          pump_electrical_energy_rate=pump_electrical_energy_rate,
      )
    else:
      raise ValueError(
          'Unsupported heat source device type:'
          f' {self._hvac.hot_water_system.heat_source_device_type}'
      )
    hws_reward_infos[hws_id] = hws_reward_info
    return hws_reward_infos, self._hvac.hot_water_system.heat_source_device_type

  def reward_info(self, occupancy_function: BaseOccupancy) -> RewardInfo:
    """Returns a message with data to compute the instantaneous reward."""
    start_time_stamp = self._current_timestamp
    end_time_stamp = start_time_stamp + pd.Timedelta(
        self._time_step_sec, unit='s'
    )

    # get zone data
    zone_reward_infos = self._get_zone_reward_infos(occupancy_function)

    # get air handler info
    air_handler_reward_infos = self._get_air_handler_reward_infos()

    # get hws info
    hws_reward_infos, hws_device_type = self._get_hws_reward_infos()

    if (
        hws_device_type
        == smart_control_building_pb2.DeviceInfo.DeviceType.BLR
    ):

      return RewardInfo(
          start_timestamp=conversion_utils.pandas_to_proto_timestamp(
              start_time_stamp
          ),
          end_timestamp=conversion_utils.pandas_to_proto_timestamp(
              end_time_stamp
          ),
          zone_reward_infos=zone_reward_infos,
          air_handler_reward_infos=air_handler_reward_infos,
          boiler_reward_infos=hws_reward_infos,
      )

    elif (
        hws_device_type
        == smart_control_building_pb2.DeviceInfo.DeviceType.ASHP
    ):
      return RewardInfo(
          start_timestamp=conversion_utils.pandas_to_proto_timestamp(
              start_time_stamp
          ),
          end_timestamp=conversion_utils.pandas_to_proto_timestamp(
              end_time_stamp
          ),
          zone_reward_infos=zone_reward_infos,
          air_handler_reward_infos=air_handler_reward_infos,
          heat_pump_reward_infos=hws_reward_infos,
      )
    else:
      raise ValueError(
          'Unsupported heat source device type:'
          f' {self._hvac.hot_water_system.heat_source_device_type}'
      )

  def step_sim(self) -> None:
    """Steps the simulation by a small amount of time.

    The following steps are completed in order to proceed to the next time step:
      1: Get external temperature.
      2: Update temperatures for each CV using FDM.
      3: Reset HVAC reheat and flow demands.
      4: For each zone/VAV:
        a: Update the VAV using the zone's temperature.
        b: Apply thermal energy from VAV to the zone.
        c: Accumulate HVAC reheat and flow demands from VAV

      Note: There is a one step delay in application of current vav
      settings/heating.
    """
    self.setup_step_sim()
    self.execute_step_sim()
