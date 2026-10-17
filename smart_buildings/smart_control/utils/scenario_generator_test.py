"""Regression checks for simulation tracking."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import pandas as pd

from smart_buildings.smart_control.proto import smart_control_reward_pb2
from smart_buildings.smart_control.utils.scenario_generator import SimulationTracker


class SimulationTrackerTest(unittest.TestCase):
    def test_step_reuses_reward_occupancy_without_sampling_vav_populations(self):
        response = smart_control_reward_pb2.RewardResponse(total_occupancy=3.5)
        occupancy = Mock()
        occupancy.average_zone_occupancy.side_effect = AssertionError(
            "Logging must not sample occupancy"
        )
        building = SimpleNamespace(
            current_timestamp=pd.Timestamp("2023-01-03 06:00"),
            occupancy=occupancy,
            simulator=SimpleNamespace(building=SimpleNamespace(
                get_zone_average_temps=lambda: {"room_0": 294.0}
            )),
        )
        result = ([0.0], -0.1, False, False, {})
        env = SimpleNamespace(
            building=building,
            step=Mock(return_value=result),
            get_reward_info_and_response=Mock(return_value=(None, response)),
        )
        # Avoid rendering setup; exercise the actual logging step.
        tracker = SimulationTracker.__new__(SimulationTracker)
        tracker.env = env
        tracker.capture_snapshots = False
        tracker.zone_ids = ["VAV_0"]
        tracker.time_step_sec = 300
        tracker.vavs = {"VAV_0": SimpleNamespace(zone_air_temperature=294.0)}
        tracker.ahus = [SimpleNamespace(
            outside_air_temperature_sensor=270.0,
            get_supply_air_temp=lambda return_temp, outside_temp: 287.0,
            run_command=1,
            air_flow_rate=2.0,
        )]
        for name in ("rewards", "timestamps", "reward_components", "occupancy",
                     "zone_temps", "ahu_log"):
            setattr(tracker, name, [])
        tracker._step_count = 0

        action = {}
        self.assertEqual(tracker.step(action), result)
        env.step.assert_called_once_with(action)
        occupancy.average_zone_occupancy.assert_not_called()
        self.assertEqual(tracker.occupancy, [3.5])
        self.assertEqual(tracker.reward_components[0]["total_occupancy"], 3.5)
        self.assertEqual(tracker._step_count, 1)


if __name__ == "__main__":
    unittest.main()
