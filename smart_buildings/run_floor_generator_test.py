"""Tests for run_floor_generator's handling of seeds with a sliver zone."""

import os
import tempfile
import unittest

os.environ.setdefault("WRAPT_DISABLE_EXTENSIONS", "true")

from smart_buildings import run_floor_generator
from smart_buildings.smart_control.utils.scenario_generator import (
    check_zone_floor_areas,
    scenario_config_from_dict,
)

# Seed 503 is the one the README example used to stop on (smallest zone
# 14.44 m2); seed 502 passes the check.
GOOD_SEED = 502
BAD_SEED = 503


def _config(seed):
    return scenario_config_from_dict(
        run_floor_generator.build_config(
            seed, f"building_{seed:04d}", "composite", None
        )
    )


class CheckZoneFloorAreasTest(unittest.TestCase):

    def test_good_seed_passes(self):
        check_zone_floor_areas(_config(GOOD_SEED))

    def test_bad_seed_raises(self):
        with self.assertRaisesRegex(ValueError, "below the 20.0 m2 minimum"):
            check_zone_floor_areas(_config(BAD_SEED))


class MainSkipsBadSeedsTest(unittest.TestCase):

    def test_no_pdf_skips_bad_seed_and_keeps_going(self):
        with tempfile.TemporaryDirectory() as out:
            run_floor_generator.main(
                ["--seeds", str(BAD_SEED), str(GOOD_SEED), "--no-pdf",
                 "--output-dir", out]
            )
            self.assertEqual(
                sorted(os.listdir(out)), [f"building_{GOOD_SEED:04d}.yaml"]
            )

    def test_all_seeds_bad_exits_nonzero(self):
        with self.assertRaises(SystemExit) as ctx:
            run_floor_generator.main(["--seeds", str(BAD_SEED), "--no-save"])
        self.assertEqual(ctx.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
