"""Tests the energy-balanced hot-water tank.

Before, the tank temperature moved toward the setpoint at a fixed rate whatever
the coils drew, while gas was billed at the setpoint. A policy could charge the
tank, drop the setpoint below the return temperature and keep drawing heat at
the tank temperature for hours with the flow term at zero: billed gas fell to
0.38 of the heat the coils delivered. Now every joule the coils take is burned,
in the same step or earlier to charge the tank.
"""

import inspect
from unittest import mock

from absl.testing import absltest
import pandas as pd

from smart_buildings.smart_control.simulator import (
    air_handler,
    boiler,
    building,
    hot_water_system,
    hvac,
    pump,
    simulator,
    simulator_flexible_floor_plan,
    weather_controller,
)
from smart_buildings.smart_control.utils import constants, test_utils

STEP_S = 300.0
CAPACITY = constants.WATER_HEAT_CAPACITY * constants.WATER_DENSITY * 1.5  # J/K


def _boiler(setpoint, energy_balance=True):
  return boiler.Boiler(
      supply_water_temperature_setpoint=setpoint,
      heating_rate=0.5,
      cooling_rate=0.1,
      energy_balance=energy_balance,
  )


def _base_simulator(tank):
  """A 2 x 2-room base Simulator with every reheat valve open."""
  air = building.MaterialProperties(50.0, 700.0, 1.2)
  wall = building.MaterialProperties(2.0, 1000.0, 1800.0)
  ext = building.MaterialProperties(0.05, 1000.0, 3000.0)
  b = building.Building(50.0, 300.0, (4, 4), (2, 2), 284.0, air, wall, ext)
  weather = weather_controller.WeatherController(265.0, 270.0)
  hws = hot_water_system.HotWaterSystem(
      heat_source=tank, pump=pump.WaterPump(6.0, 0.98)
  )
  ahu = air_handler.AirHandler(0.3, 290, 10000.0, 0.9)
  zones = list(b.get_zone_average_temps().keys())
  h = hvac.Hvac(
      zones, ahu, hws, test_utils.get_test_setpoint_schedule(), 0.45, 0.02
  )
  sim = simulator.Simulator(
      b, h, weather, STEP_S, pd.Timestamp("2021-05-10 10:00")
  )

  def open_valves():
    hws.differential_pressure_setpoint = 2.0
    for vav in h.vavs.values():
      vav.reheat_valve_setting = 1.0
      vav.damper_setting = 0.5

  return sim, b, h, ahu, weather, open_valves


class BoilerEnergyBalanceTest(absltest.TestCase):

  def test_energy_is_conserved_over_a_charge_discharge_cycle(self):
    b = _boiler(322.0)
    t0 = b.supply_water_temperature_sensor
    burned = drawn = lost = 0.0
    for k in range(200):
      b.supply_water_temperature_setpoint = 348.0 if k % 60 < 12 else 310.0
      loss = b.compute_thermal_dissipation_rate(
          b.supply_water_temperature_sensor, 280.0
      )
      b.apply_heat_draw(50_000.0, 280.0, STEP_S)
      burned += b.compute_thermal_energy_rate(0.0, 280.0, 0.0) * STEP_S
      drawn += 50_000.0 * STEP_S
      lost += loss * STEP_S
    stored = CAPACITY * (b.supply_water_temperature_sensor - t0)
    self.assertAlmostEqual(burned, drawn + lost + stored, delta=1e-6 * burned)

  def test_low_setpoint_does_not_give_free_heat(self):
    b = _boiler(348.0)
    b.supply_water_temperature_setpoint = 310.0
    t0 = b.supply_water_temperature_sensor
    b.apply_heat_draw(50_000.0, 280.0, STEP_S)
    self.assertEqual(b.compute_thermal_energy_rate(0.0, 280.0, 0.0), 0.0)
    # the tank pays for the draw: 50 kW for 5 min out of 6.3 MJ/K
    self.assertLess(b.supply_water_temperature_sensor, t0 - 2.3)

  def test_base_simulator_tank_pays_for_the_coil_heat(self):
    # Setpoint below the tank: the burner stays off, so in every step the tank
    # must lose exactly the heat the reheat coils delivered plus its standing
    # loss (the base simulator's averaged return temperature would not give it).
    tank = _boiler(340.0)
    sim, b, h, ahu, weather, open_valves = _base_simulator(tank)
    tank.supply_water_temperature_setpoint = 310.0
    for _ in range(4):
      open_valves()
      outside = weather.get_current_temp(sim._current_timestamp)
      supply_air = ahu.get_supply_air_temp(b.temp.mean(), outside)
      t_before = tank.supply_water_temperature_sensor
      loss = tank.compute_thermal_dissipation_rate(t_before, outside)
      sim.execute_step_sim()
      coil = sum(
          v.compute_reheat_energy_rate(supply_air, t_before)
          for v in h.vavs.values()
      )
      drop = CAPACITY * (t_before - tank.supply_water_temperature_sensor)
      self.assertGreater(coil, 1000.0)
      self.assertEqual(tank.compute_thermal_energy_rate(0.0, 0.0, 0.0), 0.0)
      self.assertAlmostEqual(drop / STEP_S - loss, coil, delta=1e-6 * coil)

  def test_legacy_boiler_keeps_its_sensor_read_sequence(self):
    # Each read of the legacy (setpoint-follower) tank applies another ramp
    # interval, so the base simulator must not add reads when energy_balance is
    # off: only the two per VAV that vav.output() has always made.
    tank = _boiler(320.0, energy_balance=False)
    sim, _, h, _, _, open_valves = _base_simulator(tank)
    reads = []
    original = boiler.Boiler._set_current_temperature

    def counting(self):
      reads.append(1)
      return original(self)

    open_valves()
    with mock.patch.object(boiler.Boiler, "_set_current_temperature", counting):
      sim.execute_step_sim()
    self.assertLen(reads, 2 * len(h.vavs))

  def test_both_simulators_advance_the_tank(self):
    # With energy_balance on, the boiler no longer moves its own temperature on
    # a sensor read, so a simulator that skipped apply_heat_draw would freeze
    # the tank and bill no gas at all.
    for step in (
        simulator.Simulator.execute_step_sim,
        simulator_flexible_floor_plan.SimulatorFlexibleGeometries.execute_step_sim,
    ):
      self.assertIn(
          "hot_water_system.apply_heat_draw(", inspect.getsource(step)
      )


if __name__ == "__main__":
  absltest.main()
