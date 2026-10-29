"""Checks the air handlers' return-air energy balance, step by step, without changing the simulator.

Each step, for every air handler, this compares the recirculation temperature the simulator uses with the airflow-
weighted mixture of the air its VAVs extract, and reports:

  imbalance_W  r * rho * c_p * V_total * (T_used - T_mixture): heat (+) or cooling (-) that no air stream carried
  passthru_W   the part of it that reaches the zones (mixed air passes the cooling coil unchanged)
  S_mix_WK     entropy generated in the mixing box by the actual streams (recirculated extract at each VAV's zone
               temperature plus outdoor air) ending at the mixed temperature the simulator uses; < 0 is impossible

Usage, from the starter-kit root:
  python smart_buildings/check_return_air.py --config smart_buildings/training_scenarios/<building>/<weather>/config.gin
      [--policy pid | high-flow-cold] [--steps 576] [--out check.csv]

'high-flow-cold' is a scripted stress test on top of the PID baseline: air handlers on at maximum static pressure,
recirculation and supply temperature setpoint, dampers fully open on the colder half of each handler's zones and closed
on the warmer half, reheat valves closed. On a correct simulator imbalance_W is 0 (to rounding) for every policy.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from smart_buildings import evaluate
from smart_buildings.baselines.pid import PIDPolicy
from smart_buildings.smart_control.simulator import air_handler as air_handler_py
from smart_buildings.smart_control.simulator import constants

RHO_CP = constants.AIR_DENSITY * constants.AIR_HEAT_CAPACITY


class HighFlowCold(PIDPolicy):
  """Stress test: airflow to the colder half of each air handler's zones only, no reheat."""

  def get_action(self, obs):
    act = super().get_action(obs)
    c, d = act['continuous_action'], act['discrete_action']
    for ctrl in self.ahu_controllers:
      d[ctrl['run_discrete_idx']] = 1
      if ctrl['pressure_idx'] is not None:
        c[ctrl['pressure_idx']] = 1.0
      c[ctrl['temp_idx']] = 1.0
      zones = sorted(ctrl['zones'], key=lambda z: self._denormalize_zone_temp(obs[z[1]]))
      for k, (_, _, idx) in enumerate(zones):
        damper = idx.get('supply_air_damper_percentage_command')
        reheat = idx.get('reheat_valve_percentage_command')
        if damper is not None:
          c[damper] = 1.0 if k < len(zones) // 2 else -1.0
        if reheat is not None:
          c[reheat] = -1.0
    for i in self.recirculation_indices:
      c[i] = 1.0
    return act


POLICIES = {'pid': PIDPolicy, 'high-flow-cold': HighFlowCold}


class ReturnAirCheck:
  """Records the return-air balance while the simulator computes the supply air temperature."""

  def __init__(self, sim):
    self.sim, self.rows, self._in_step = sim, [], False
    hvac = sim.hvac
    self.ahus = (hvac.air_handler.ahus
                 if isinstance(hvac.air_handler, air_handler_py.AirHandlerSystem)
                 else [hvac.air_handler])
    step, supply = sim.execute_step_sim, hvac.air_handler.get_supply_air_temp

    def execute_step_sim(*a, **k):
      self._in_step = True
      try:
        return step(*a, **k)
      finally:
        self._in_step = False

    def get_supply_air_temp(recirculation, ambient):
      out = supply(recirculation, ambient)
      if self._in_step:   # the bill calls it too; record only the physics step
        self._record(recirculation, ambient, out)
      return out

    sim.execute_step_sim = execute_step_sim
    hvac.air_handler.get_supply_air_temp = get_supply_air_temp

  def _record(self, recirculation, ambient, supply):
    """Called inside the step, after the dampers are set and before the zones are solved."""
    sim, hvac = self.sim, self.sim.hvac
    temps = sim.building.get_zone_average_temps()
    areas = {z: len(c) for z, c in sim.building.room_dict.items()}
    row = dict(imbalance_W=0.0, passthru_W=0.0, S_mix_WK=0.0, S_mix_negative_ahus=0)
    for ahu in self.ahus:
      ahu_id = ahu.device_id()
      t_used = recirculation[ahu_id] if isinstance(recirculation, dict) else recirculation
      t_sup = supply[ahu_id] if isinstance(supply, dict) else supply
      streams = []   # (airflow m3/s, extract temperature K) of each VAV that delivers air
      for v_id, vav in hvac.vavs.items():
        if vav.air_handler.device_id() != ahu_id or not (vav.damper_setting > 0 and vav.max_air_flow_rate > 0):
          continue
        zones = [z for z in hvac.get_zones_for_vav(v_id) if z in temps]
        area = sum(areas[z] for z in zones)
        if area > 0:
          streams.append((vav.unrestricted_flow_rate_demand, sum(temps[z] * areas[z] for z in zones) / area))
      if not streams:
        continue
      flow, temp = np.array(streams).T
      flow = flow * min(1.0, ahu.max_air_flow_rate / flow.sum())   # the handler's cap, common to its VAVs
      g, r = RHO_CP * flow, ahu.recirculation
      t_mixture = float(g @ temp / g.sum())
      t_mix = r * t_used + (1 - r) * ambient
      imbalance = r * g.sum() * (t_used - t_mixture)
      s_mix = r * float(g @ np.log(t_mix / temp)) + (1 - r) * g.sum() * np.log(t_mix / ambient)
      scale = (r * float(g @ np.abs(t_mix - temp)) + (1 - r) * g.sum() * abs(t_mix - ambient)) / t_mix
      row['imbalance_W'] += imbalance
      row['passthru_W'] += imbalance if np.isclose(t_sup, t_mix) else 0.0
      row['S_mix_WK'] += s_mix
      row['S_mix_negative_ahus'] += int(s_mix < -1e-6 * scale)
    self.rows.append(row)


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('--config', type=Path, required=True)
  p.add_argument('--policy', choices=sorted(POLICIES), default='pid')
  p.add_argument('--steps', type=int, default=576)
  p.add_argument('--out', type=Path)
  a = p.parse_args()

  env = evaluate.load_hybrid_environment(a.config)
  env.metrics_path = None   # no metrics files from a check run
  meta = evaluate.create_building_meta_data(env, a.config.parent)
  policy = evaluate.initialize_policy(POLICIES[a.policy], meta, env.action_spec())
  obs, _ = env.reset()
  check = ReturnAirCheck(env.building.simulator)
  for _ in range(a.steps):
    action = evaluate.validate_action(evaluate.get_policy_action(policy, obs), env.action_spec())
    obs, _, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
      break

  rows = check.rows
  hours = env.building.simulator.time_step_sec / 3600
  imb = np.array([r['imbalance_W'] for r in rows])
  pt = np.array([r['passthru_W'] for r in rows])
  neg = np.array([r['S_mix_negative_ahus'] for r in rows])
  print(f'{a.config.parent.parent.name}/{a.config.parent.name}, policy {a.policy}, {len(rows)} steps')
  print(f'  return-air imbalance: mean {imb.mean() / 1e3:+.2f} kW, min {imb.min() / 1e3:+.2f}, max {imb.max() / 1e3:+.2f}')
  print(f'  reaching the zones: {pt.sum() * hours / 1e6:+.2f} MWh in total, mean {pt.mean() / 1e3:+.2f} kW')
  print(f'  steps with negative mixing-box entropy generation: {100 * (neg > 0).mean():.1f} %')
  if a.out:
    with open(a.out, 'w', newline='') as f:
      w = csv.DictWriter(f, list(rows[0]))
      w.writeheader()
      w.writerows(rows)


if __name__ == '__main__':
  main()
