"""The return air an air handler recirculates is the airflow-weighted mixture of its VAVs' extract.

With unequal airflow per unit floor area, an area-weighted average does not carry the enthalpy of the air that left
the zones: the handler then delivers heat (or cooling) that no stream carried.
"""

from pathlib import Path
import traceback

import numpy as np
from absl.testing import absltest

from smart_buildings.evaluate import load_hybrid_environment

_SCENARIO_ROOT = Path('smart_buildings/training_scenarios')


def _first_scenario() -> Path:
  building = sorted(p for p in _SCENARIO_ROOT.iterdir() if p.is_dir())[0]
  strip = sorted(p for p in building.iterdir() if p.is_dir())[0]
  return strip / 'config.gin'


class ReturnAirMixingTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.env = load_hybrid_environment(_first_scenario())

  def setUp(self):
    super().setUp()
    self.env.reset()
    self.sim = self.env.building.simulator
    hvac = self.sim.hvac
    by_ahu = {}
    for v_id, vav in hvac.vavs.items():
      if hvac.get_zones_for_vav(v_id):
        by_ahu.setdefault(vav.air_handler.device_id(), []).append(v_id)
    # An air handler with at least two VAVs: the first one cold, the others warm.
    self.ahu_id, self.v_ids = next((a, v) for a, v in by_ahu.items() if len(v) >= 2)
    for v_id in self.v_ids:
      for zone in hvac.get_zones_for_vav(v_id):
        for r, c in self.sim.building.room_dict[zone]:
          self.sim.building.temp[r, c] = 283.15 if v_id == self.v_ids[0] else 298.15

  def _vav_means(self):
    """(area, temperature, mixing weight) of each VAV of the chosen handler, from the zone fields."""
    hvac, rooms = self.sim.hvac, self.sim.building.room_dict
    avg = self.sim.building.get_zone_average_temps()
    out = []
    for v_id in self.v_ids:
      vav = hvac.vavs[v_id]
      zones = hvac.get_zones_for_vav(v_id)
      area = sum(len(rooms[z]) for z in zones)
      temp = sum(avg[z] * len(rooms[z]) for z in zones) / area
      flow = (
          vav.unrestricted_flow_rate_demand
          if vav.damper_setting > 0 and vav.max_air_flow_rate > 0
          else 0.0
      )
      out.append((area, temp, flow))
    return avg, out

  def test_recirculated_air_is_the_airflow_weighted_mixture(self):
    hvac = self.sim.hvac
    for v_id in self.v_ids:   # cold VAV at full airflow, the others at their minimum
      vav = hvac.vavs[v_id]
      vav.damper_setting = 1.0 if v_id == self.v_ids[0] else vav.min_damper_setting
    avg, vavs = self._vav_means()
    flow_mix = sum(f * t for _, t, f in vavs) / sum(f for _, _, f in vavs)
    area_mix = sum(a * t for a, t, _ in vavs) / sum(a for a, _, _ in vavs)

    got = self.sim._recirculation_temps(avg)[self.ahu_id]  # pylint: disable=protected-access
    self.assertAlmostEqual(got, flow_mix, places=9)
    # Not vacuous: the old area weighting gives a clearly warmer return here.
    self.assertGreater(area_mix - got, 0.5)

  def test_closed_dampers_carry_no_weight(self):
    hvac = self.sim.hvac
    for v_id in self.v_ids:   # only the cold VAV moves air
      hvac.vavs[v_id].damper_setting = 1.0 if v_id == self.v_ids[0] else 0.0
    avg, vavs = self._vav_means()
    got = self.sim._recirculation_temps(avg)[self.ahu_id]  # pylint: disable=protected-access
    self.assertAlmostEqual(got, vavs[0][1], places=9)

  def test_no_airflow_falls_back_to_the_area_mean(self):
    for v_id in self.v_ids:
      self.sim.hvac.vavs[v_id].damper_setting = 0.0
    avg, vavs = self._vav_means()
    area_mix = sum(a * t for a, t, _ in vavs) / sum(a for a, _, _ in vavs)
    got = self.sim._recirculation_temps(avg)[self.ahu_id]  # pylint: disable=protected-access
    self.assertAlmostEqual(got, area_mix, places=9)

  def test_the_step_and_the_cooling_bill_both_use_it(self):
    calls = []
    original = self.sim._recirculation_temps  # pylint: disable=protected-access

    def spy(avg_temps):
      calls.append(traceback.extract_stack()[-2].name)
      return original(avg_temps)

    self.sim._recirculation_temps = spy  # pylint: disable=protected-access
    try:
      spec = self.env.action_spec()
      self.env.step({
          'continuous_action': np.ones(spec['continuous_action'].shape, dtype=np.float32),
          'discrete_action': np.ones(spec['discrete_action'].shape, dtype=np.int32),
      })
    finally:
      del self.sim._recirculation_temps  # pylint: disable=protected-access
    self.assertCountEqual(
        calls, ['execute_step_sim', '_get_air_handler_reward_infos']
    )


if __name__ == '__main__':
  absltest.main()
