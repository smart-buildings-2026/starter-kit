#!/usr/bin/env python3
"""Generate building floor-plan configs for the competition.

Each building is described by a small YAML config (footprint shape and size,
number of floors, room size, number of AHUs, occupants per zone). The
simulator turns that config into a full floor plan, zone map, and HVAC
topology at scenario-generation time (see ``run_scenario_generator.py``).

You can sample arbitrarily many additional buildings here to enlarge your
training distribution. The evaluation buildings are drawn from the same
generator and the same parameter ranges.

Each building gets its own envelope construction as well as its own geometry,
written into the YAML as a ``materials:`` block. Six properties are drawn per
building, the same way the shipped eight were drawn; see
``smart_control/utils/materials.py`` for the ranges and for why they are drawn
independently. Pass ``--no-materials`` to leave the block out and fall back to
the default construction.

Materials are stratified across the batch, so the same command always
reproduces the same set, while a different ``--count`` redraws. Geometry
depends only on the building's own seed.

Usage:
    # Preview 3 buildings without saving (opens matplotlib windows)
    python -m smart_buildings.run_floor_generator --count 3 --visualize --no-save

    # Save 10 new buildings (seeds 500-509) as YAML + PDF previews
    python -m smart_buildings.run_floor_generator \
        --start-seed 500 --count 10 --output-dir smart_buildings/buildings

    # Regenerate one building from an explicit seed, without the PDF
    python -m smart_buildings.run_floor_generator --seeds 500 --no-pdf
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from smart_buildings.smart_control.utils.config_generator import (
    BUILDING_SHAPES,
    COMPOSITE_STYLES,
    generate_config,
    visualize_config,
)
from smart_buildings.smart_control.utils.materials import (
    draw_materials,
    envelope_capacity,
    facade_u,
)
from smart_buildings.smart_control.utils.scenario_generator import (
    check_zone_floor_areas,
    list_recorded_weather_presets,
    scenario_config_from_dict,
)

# Sampling ranges used for the shipped development buildings. Each range is
# sampled deterministically from the building's seed, so a given seed always
# produces the same building.
DEFAULT_RANGES = dict(
    width=(400, 700),
    height=(300, 500),
    num_floors=(2, 4),
    min_room_size=(30, 60),
    # max_room_size is min_room_size plus this. The shipped buildings span 11
    # to 16, so generated rooms are shaped the same way.
    room_size_span=(11, 16),
    shape_ratio=(0.28, 0.40),
    composite_coverage=(0.65, 0.85),
    composite_style="architectural",
    # Inclusive on both ends: config_generator draws with random.randint. This
    # is a floor, not the final count -- the scenario generator raises it to
    # ceil(zones / max_zones_per_ahu) when the floor plan needs more handlers,
    # so a large footprint can end up with more air handlers than this.
    num_ahus=(3, 7),
    occupants_per_zone=(1, 2),
)


def build_config(
    seed: int,
    name: str,
    building_shape: str,
    materials: dict[str, float] | None = None,
) -> dict:
    """Generate one building config dict for a given seed.

    Geometry comes from the seed alone. Materials are passed in, because they
    are stratified across the whole batch rather than drawn per building; see
    utils/materials.py.
    """
    config = generate_config(
        name=name,
        seed=seed,
        building_shape=building_shape,
        **DEFAULT_RANGES,
    )
    if materials is not None:
        config["materials"] = materials
    return config


def summarize(config: dict) -> None:
    fp = config["floor_plan"]
    line = (
        f"  {config['name']}: shape={fp['building_shape']} "
        f"{fp['width']}x{fp['height']} ft, floors={fp['num_floors']}, "
        f"AHUs>={config['ahu']['num_ahus']}"
    )
    m = config.get("materials")
    if m:
        line += f", facade U={facade_u(m):.2f} W/m2K, envelope C={envelope_capacity(m)/1e6:.2f} MJ/m3K"
    print(line)


def save_config(
    config: dict, output_dir: Path, save_pdf: bool, show: bool
) -> list[Path]:
    """Write a building YAML (and optional PDF preview); return written paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    yaml_path = output_dir / f"{config['name']}.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    written.append(yaml_path)
    print(f"    saved {yaml_path}")

    if save_pdf:
        pdf_path = output_dir / f"{config['name']}.pdf"
        visualize_config(config, save_path=str(pdf_path), show=show)
        written.append(pdf_path)
        print(f"    saved {pdf_path}")

    return written


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Explicit list of seeds to generate (overrides --start-seed/--count).",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=500,
        help="First seed when --seeds is not given (default: 500, chosen to sit "
        "well clear of the seeds behind the shipped buildings).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of buildings to generate from --start-seed (default: 5).",
    )
    parser.add_argument(
        "--shape",
        default="composite",
        choices=BUILDING_SHAPES,
        help="Footprint shape (default: composite, as used by the shipped set).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("smart_buildings/buildings"),
        help="Where to write building_XXXX.yaml files.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Generate and (optionally) preview only; do not write files.",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip the PDF floor-plan preview when saving.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open an interactive matplotlib preview for each building.",
    )
    parser.add_argument(
        "--no-materials",
        action="store_true",
        help="Do not write a materials block, so buildings fall back to the "
        "default construction and differ only in geometry.",
    )
    return parser


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)

    print("Available shapes:", BUILDING_SHAPES)
    print("Available composite styles:", COMPOSITE_STYLES)
    # Deliberately NOT CLIMATES. That dict holds the legacy synthetic sinusoid
    # presets, every one of which load_weather_config refuses, so listing it
    # here would offer names that cannot be generated from. Buildings are
    # weather-agnostic anyway -- the weather is chosen later, by
    # run_scenario_generator.
    print("Available weather presets:", list_recorded_weather_presets())
    print()

    seeds = (
        args.seeds
        if args.seeds
        else list(range(args.start_seed, args.start_seed + args.count))
    )
    print(
        f"Generating {len(seeds)} building(s) with shape='{args.shape}': seeds {seeds}"
    )

    # Materials are stratified across the batch rather than drawn per seed, so
    # that even a handful of buildings covers each property's range instead of
    # clustering. The RNG is seeded from the seed list, so repeating the same
    # command reproduces the same set exactly.
    if args.no_materials:
        materials = [None] * len(seeds)
    else:
        # Hash the seed list to an int rather than seeding Random with the
        # tuple directly, so the draw does not depend on Python's hashing.
        digest = hashlib.sha256(repr(seeds).encode()).digest()
        materials = draw_materials(
            random.Random(int.from_bytes(digest[:8], "big")), len(seeds)
        )

    written: list[Path] = []
    skipped: list[int] = []
    for seed, mats in zip(seeds, materials):
        config = build_config(seed, f"building_{seed:04d}", args.shape, mats)
        summarize(config)
        # Check every seed, not only the ones that get a PDF preview: the
        # preview used to be the only place the floor plan was generated, so
        # --no-pdf and --no-save let a sliver zone through to be found later,
        # when a scenario is built from the YAML. A bad seed is skipped rather
        # than ending the run, because a sizeable share of seeds produce one.
        try:
            check_zone_floor_areas(scenario_config_from_dict(config))
        except ValueError as err:
            print(f"    skipped seed {seed}: {err}")
            skipped.append(seed)
            continue
        if args.visualize and args.no_save:
            visualize_config(config, show=True)
        if not args.no_save:
            written += save_config(
                config, args.output_dir, save_pdf=not args.no_pdf, show=args.visualize
            )

    kept = len(seeds) - len(skipped)
    if args.no_save:
        print(f"\nDone (preview only). Generated {kept} building config(s).")
    else:
        print(f"\nDone. Wrote {len(written)} file(s) to {args.output_dir}.")
    if skipped:
        print(
            f"Skipped {len(skipped)} seed(s) whose floor plan has a zone below "
            f"the minimum floor area: {skipped}. Pick other seeds to replace them."
        )
    if kept == 0:
        sys.exit(1)


if __name__ == "__main__":
    os.environ.setdefault("WRAPT_DISABLE_EXTENSIONS", "true")
    main()
