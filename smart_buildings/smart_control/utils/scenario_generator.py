"""Scenario generator for HVAC simulations.

Generates complete simulation scenarios from YAML configuration files.
Each scenario includes floor plans, zone maps, and gin configs in a named directory.

Example usage:
    from smart_buildings.smart_control.utils.scenario_generator import (
        generate_scenario, get_env, print_action_spec, SimulationTracker
    )

    # Generate scenario and create env
    result = generate_scenario("path/to/config.yaml")
    env = get_env(result)

    # See action spec
    action = print_action_spec(env)

    # Run simulation with tracking
    tracker = SimulationTracker(env)

    for i in range(1000):
        tracker.step(action)

        if i % 100 == 0:
            tracker.render()      # Show building visualization
            tracker.plot()        # Show all charts

    # Render a specific past timestep
    tracker.render(step=500)

    # Get summary
    tracker.summary()
"""

import os
import random
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import yaml

# Gin registration imports. Kept even though nothing below references them:
# gin resolves configurables by name when it parses a config.gin, so the
# @gin.configurable decorators have to have run first.
from smart_buildings.smart_control.reward import electricity_energy_cost  # noqa: F401
from smart_buildings.smart_control.reward import natural_gas_energy_cost  # noqa: F401
from smart_buildings.smart_control.reward import setpoint_energy_carbon_regret  # noqa: F401
from smart_buildings.smart_control.simulator import air_handler  # noqa: F401
from smart_buildings.smart_control.simulator import boiler  # noqa: F401
from smart_buildings.smart_control.simulator import building  # noqa: F401
from smart_buildings.smart_control.simulator import hot_water_system  # noqa: F401
from smart_buildings.smart_control.simulator import hvac  # noqa: F401
from smart_buildings.smart_control.simulator import hvac_floorplan_based  # noqa: F401
from smart_buildings.smart_control.simulator import pump  # noqa: F401
from smart_buildings.smart_control.simulator import randomized_arrival_departure_occupancy  # noqa: F401
from smart_buildings.smart_control.simulator import setpoint_schedule  # noqa: F401
from smart_buildings.smart_control.simulator import simulator_building  # noqa: F401
from smart_buildings.smart_control.simulator import simulator_flexible_floor_plan  # noqa: F401
from smart_buildings.smart_control.simulator import stochastic_convection_simulator  # noqa: F401
from smart_buildings.smart_control.simulator import weather_controller  # noqa: F401
from smart_buildings.smart_control.utils import environment_utils  # noqa: F401
from smart_buildings.smart_control.utils import observation_normalizer  # noqa: F401

from smart_buildings.smart_control.utils.floor_generator import (
    OBSERVATION_NORMALIZER_CONFIG,
    BSPFloorPlan,
    MultiFloorPlan,
    assign_rooms_to_ahus,
    required_ahus,
)

# Lock to prevent race conditions when multiple threads generate floor plans
# with seeded random number generators. Without this lock, concurrent threads
# can interleave their random calls, causing different floor plans to be
# generated even with the same seed.
_floor_plan_generation_lock = threading.Lock()


# Control volume edge length in cm, and the floor-to-ceiling height, both as
# written into every generated config.gin. They are constants here rather than
# literals in the template because the zone floor area check below has to agree
# with what the simulator will actually read; a literal in each place is how the
# two drift apart.
CONTROL_VOLUME_CM = 20
FLOOR_HEIGHT_CM = 300.0

# Smallest floor area, in m2, a generated zone may have. Matches the default in
# FloorPlanBasedBuilding, which refuses to construct a building that violates
# it. See _check_generated_zone_areas for why the limit exists and why 20.
MIN_ZONE_FLOOR_AREA_M2 = 20.0


def _check_generated_zone_areas(generator, cv_size_cm: int, seed) -> None:
    """Raises if the floor plan just generated carries a zone that is a sliver.

    `min_room_size` bounds the BSP leaves, but the composite shape mask is
    applied after the split and clips any leaf crossing its boundary. The
    surviving connected component is a size nothing checked, so the realised
    minimum cannot be set from the floor plan parameters: building_train_5 and
    building_train_7 share min_room_size 25 and composite_coverage 0.67 and come
    out with smallest zones of 9.36 m2 and 50.88 m2.

    Such a zone diverges rather than merely looking wrong. Occupant heat is
    assigned per zone while internal gains, the VAV's airflow share and the mean
    temperature its thermostat reads all scale with floor area, so a small zone
    takes a full zone's worth of occupant heat into very little air while
    contributing almost nothing to the signal the loop controlling it can see.
    building_train_8's 9.88 m2 room_1 reached 149.2 C before this check
    existed, bounded only by the solver's own temperature clip.

    The limit is 20 m2 because the runaway stops by about 17 m2 in measurement,
    and because 20 m2 is 10 m2 per occupant at the shipped zone assignment of 2,
    an ordinary office density. Keeping zones that size is what makes per-zone
    occupancy a sound assumption, so this is a geometry check rather than a
    change to the occupancy model.

    Args:
        generator: A generated BSPFloorPlan or MultiFloorPlan.
        cv_size_cm: Control volume edge length, in cm.
        seed: The seed that produced this plan, named in the error so the fix is
          obvious.

    Raises:
        ValueError: if any zone is below MIN_ZONE_FLOOR_AREA_M2.
    """
    if MIN_ZONE_FLOOR_AREA_M2 <= 0.0:
        return
    labels, _ = generator.get_room_labels()
    cv_area_m2 = (cv_size_cm / 100.0) ** 2
    counts = np.bincount(np.asarray(labels).ravel())
    # Label 0 is background in connectedComponents, so it is not a zone.
    areas = [
        (i, counts[i] * cv_area_m2)
        for i in range(1, len(counts))
        if counts[i] > 0
    ]
    offenders = sorted(
        (a for a in areas if a[1] < MIN_ZONE_FLOOR_AREA_M2), key=lambda a: a[1]
    )
    if not offenders:
        return
    listed = ", ".join(f"room_{i} {area:.2f} m2" for i, area in offenders)
    smallest = min(a[1] for a in areas)
    raise ValueError(
        f"seed {seed} produces {len(offenders)} zone(s) below the "
        f"{MIN_ZONE_FLOOR_AREA_M2} m2 minimum floor area (smallest "
        f"{smallest:.2f} m2): {listed}. A zone this small is a floor plan "
        "artifact, not a room, and it diverges in simulation. min_room_size "
        "does not control the realised minimum, so change the seed and measure "
        "again rather than raising min_room_size."
    )


@dataclass
class FloorPlanConfig:
    """Floor plan generation parameters."""

    width: int = 100
    height: int = 100
    num_floors: int = 1
    min_room_size: int = 20
    max_room_size: int = 40
    wall_thickness: int = 1
    door_width: int = 2
    split_variance: float = 0.3
    building_shape: str = "rectangle"
    shape_ratio: float = 0.33
    composite_coverage: float = 0.5
    composite_style: str = "architectural"


from smart_buildings.smart_control.reward.energy_scale import (
    C_ELECTRICITY_PER_AHU,
    C_NATURAL_GAS_PER_AHU,
)

# Most air one VAV box can draw, m3/s. The AHU fan is sized from this and the
# zones-per-AHU cap, so the two can never disagree.
VAV_MAX_AIR_FLOWRATE = 2.0


@dataclass
class MaterialsConfig:
    """Construction of one building: envelope, partitions, air, thermal mass.

    Buildings that share a construction differ only in shape, and shape alone
    moves the control problem far less than it appears to: with an identical
    envelope the free-float equilibrium above ambient is `internal gain / UA`,
    which is the same everywhere. Varying construction is what makes two
    buildings genuinely different to control.

    Only `density * heat_capacity` reaches the solver, as a volumetric heat
    capacity. Conductivities are effective values for a control volume, not
    handbook values for a bulk material. One CV stands in for a whole facade,
    so heat crosses half of it and then a surface film:

        1/U = 0.1 / k_exterior + 0.299

    which is what anchors the named constructions below to a real facade U.
    """

    exterior_cv_conductivity: float = 0.21
    exterior_cv_density: float = 2200.0
    exterior_cv_heat_capacity: float = 910.0

    interior_wall_cv_conductivity: float = 0.5
    interior_wall_cv_density: float = 160.0
    interior_wall_cv_heat_capacity: float = 900.0

    interior_cv_conductivity: float = 50.0
    interior_cv_density: float = 1.2
    interior_cv_heat_capacity: float = 1005.0

    interior_mass_cv_conductivity: float = 15.0
    interior_mass_cv_density: float = 23.0
    interior_mass_cv_heat_capacity: float = 1000.0

    @property
    def facade_u_value(self) -> float:
        """Whole-building effective facade U in W/m2K, for the config header."""
        return 1.0 / (0.1 / self.exterior_cv_conductivity + 0.299)

    @property
    def envelope_volumetric_capacity(self) -> float:
        """Facade CV volumetric heat capacity in MJ/m3K."""
        return self.exterior_cv_density * self.exterior_cv_heat_capacity / 1e6


# Named constructions. The two axes that matter are how fast heat leaves
# (facade U) and how much heat the fabric holds (mass), because together they
# set the thermal time constant and the free-float equilibrium -- the two
# things that decide whether a climate is easy or hard for a controller.
#
# The interior air node is never varied: it is room air, and room air is room
# air in every building.
CONSTRUCTIONS: dict[str, MaterialsConfig] = {
    # Thick masonry, punched windows. Slow, forgiving, long memory.
    "heavy_masonry": MaterialsConfig(
        exterior_cv_conductivity=0.15,
        exterior_cv_density=2400.0,
        exterior_cv_heat_capacity=1000.0,
        interior_wall_cv_conductivity=0.45,
        interior_wall_cv_density=700.0,
        interior_wall_cv_heat_capacity=900.0,
        interior_mass_cv_conductivity=15.0,
        interior_mass_cv_density=40.0,
    ),
    # The default construction: the values MaterialsConfig itself carries.
    "medium_concrete": MaterialsConfig(),
    # Metal stud and insulated panel. Light, responds quickly.
    "light_steel_frame": MaterialsConfig(
        exterior_cv_conductivity=0.30,
        exterior_cv_density=600.0,
        exterior_cv_heat_capacity=900.0,
        interior_wall_cv_conductivity=0.55,
        interior_wall_cv_density=120.0,
        interior_wall_cv_heat_capacity=900.0,
        interior_mass_cv_conductivity=15.0,
        interior_mass_cv_density=14.0,
    ),
    # Highly glazed curtain wall. Leaky and almost massless: the hardest case,
    # because it tracks the weather rather than riding through it.
    "curtain_wall": MaterialsConfig(
        exterior_cv_conductivity=0.55,
        exterior_cv_density=350.0,
        exterior_cv_heat_capacity=840.0,
        interior_wall_cv_conductivity=0.60,
        interior_wall_cv_density=110.0,
        interior_wall_cv_heat_capacity=900.0,
        interior_mass_cv_conductivity=15.0,
        interior_mass_cv_density=10.0,
    ),
}


@dataclass
class AHUConfig:
    """AHU (Air Handling Unit) configuration."""

    num_ahus: int = 2
    # None means "size the fan to the cap", which is almost always what you
    # want: see max_zones_per_ahu below. Set a number to override.
    max_air_flow_rate: float | None = None
    # Most zones one air handler may serve. `num_ahus` above is a floor, not
    # the answer: if the floor plan has more rooms than `num_ahus` handlers can
    # serve at this cap, more handlers are added.
    #
    # The cap and the fan size have to move together, and that is why
    # `max_air_flow_rate` defaults to None. One VAV can draw at most
    # `vav_max_air_flowrate`, so a handler serving `n` zones needs
    # `n * vav_max_air_flowrate` to satisfy all of them at once. A fan sized
    # independently of the cap leaves handlers short of air whenever their
    # zones call together, and how short depends on how many zones the floor
    # plan happened to give them.
    #
    # Note that capping zones per handler is not on its own enough: it bounds
    # the demand but says nothing about the supply. Deriving the fan from the
    # cap makes the two consistent by construction, at any cap.
    max_zones_per_ahu: int = 10

    @property
    def sized_max_air_flow_rate(self) -> float:
        """Fan size actually used: explicit if given, else derived from the cap."""
        if self.max_air_flow_rate is not None:
            return self.max_air_flow_rate
        return VAV_MAX_AIR_FLOWRATE * self.max_zones_per_ahu
    recirculation_ratio: float = 0.3
    supply_air_temperature_setpoint: float = 298.0
    fan_static_pressure: float = 1000.0
    fan_efficiency: float = 0.9


@dataclass
class WeatherConfig:
    """Weather/ambient temperature configuration."""

    high_temp: float = 305.0  # Kelvin
    low_temp: float = 280.0  # Kelvin
    convection_coefficient: float = 100.0
    # Real recorded weather. When weather_csv is set the sinusoid is replaced by
    # ReplayWeatherController reading that file, and high_temp/low_temp are
    # ignored. The path is resolved relative to the weather preset directory.
    weather_csv: str = ""
    # A replay file covers a whole year, so the scenario has to start in the
    # month the preset is about. Empty means keep the building's own timestamp.
    start_timestamp: str = ""


@dataclass
class HotWaterSystemConfig:
    """Hot water system configuration."""

    pump_differential_head: float = 6.0
    pump_efficiency: float = 0.98
    supply_water_temperature_setpoint: float = 340.0
    boiler_heating_rate: float = 0.5
    boiler_cooling_rate: float = 0.1


@dataclass
class SetpointScheduleConfig:
    """Temperature setpoint schedule configuration."""

    morning_start_hour: int = 6
    evening_start_hour: int = 19
    heating_setpoint_day: float = 294.0
    cooling_setpoint_day: float = 297.0
    heating_setpoint_night: float = 289.0
    cooling_setpoint_night: float = 298.0
    time_zone: str = "US/Pacific"


@dataclass
class RewardConfig:
    """Reward function configuration."""

    max_productivity_personhour_usd: float = 300.0
    min_productivity_personhour_usd: float = 0.0
    productivity_deviation_ref: float = 5.0
    # Energy-normalisation denominators, expressed per air handler. The gin
    # emits c * num_ahus. A single flat value gave a per-AHU allowance that
    # varied with building size for no reason (2.33x across the ten shipped
    # buildings for gas), so identical physical draw scored differently
    # depending on how many air handlers a building happened to have.
    # Imported, never duplicated. These live in reward/energy_scale.py, which
    # check_scales.py and the other generator also read, so every consumer sees
    # one definition and the copies cannot drift apart.
    c_electricity_per_ahu: float = C_ELECTRICITY_PER_AHU
    c_natural_gas_per_ahu: float = C_NATURAL_GAS_PER_AHU
    productivity_weight: float = 0.6
    energy_cost_weight: float = 0.2
    carbon_emission_weight: float = 0.2


@dataclass
class SimulationConfig:
    """Simulation timing configuration."""

    time_step_sec: int = 300
    internal_gain_w_per_m2: float = 10.0
    internal_gain_standby_fraction: float = 0.2
    start_timestamp: str = "2023-07-10 06:00"  # Monday
    num_days_in_episode: int = 30
    discount_factor: float = 0.9
    occupants_per_zone: int = 1  # Number of occupants per zone


@dataclass
class ScenarioConfig:
    """Complete scenario configuration."""

    name: str = "default_scenario"
    seed: int | None = None
    output_base_dir: str = "scenarios"
    write_metrics_dir: bool = True

    floor_plan: FloorPlanConfig = field(default_factory=FloorPlanConfig)
    ahu: AHUConfig = field(default_factory=AHUConfig)
    # Name from CONSTRUCTIONS, or "custom" when `materials` is given directly.
    # Defaults to the construction every scenario shipped with, so a config
    # that says nothing about materials generates byte-identically to before.
    construction: str = "medium_concrete"
    materials: MaterialsConfig | None = None
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    hot_water_system: HotWaterSystemConfig = field(default_factory=HotWaterSystemConfig)
    setpoint_schedule: SetpointScheduleConfig = field(
        default_factory=SetpointScheduleConfig
    )
    reward: RewardConfig = field(default_factory=RewardConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)


def load_config(config_path: str) -> ScenarioConfig:
    """Load scenario configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        ScenarioConfig object with all parameters.
    """
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    config = ScenarioConfig()

    # Top-level fields
    if "name" in raw:
        config.name = raw["name"]
    if "seed" in raw:
        config.seed = raw["seed"]
    if "output_base_dir" in raw:
        config.output_base_dir = raw["output_base_dir"]
    if "write_metrics_dir" in raw:
        config.write_metrics_dir = raw["write_metrics_dir"]
    if "construction" in raw:
        config.construction = raw["construction"]
    if "materials" in raw:
        config.materials = MaterialsConfig(**raw["materials"])

    # Nested configs
    if "floor_plan" in raw:
        config.floor_plan = FloorPlanConfig(**raw["floor_plan"])
    if "ahu" in raw:
        config.ahu = AHUConfig(**raw["ahu"])
    if "weather" in raw:
        config.weather = WeatherConfig(**raw["weather"])
    if "hot_water_system" in raw:
        config.hot_water_system = HotWaterSystemConfig(**raw["hot_water_system"])
    if "setpoint_schedule" in raw:
        config.setpoint_schedule = SetpointScheduleConfig(**raw["setpoint_schedule"])
    if "reward" in raw:
        config.reward = RewardConfig(**raw["reward"])
    if "simulation" in raw:
        config.simulation = SimulationConfig(**raw["simulation"])

    return config


def scenario_config_from_dict(config_dict: dict[str, Any]) -> ScenarioConfig:
    """Build a ScenarioConfig from a dict matching the YAML structure."""
    config = ScenarioConfig()

    if "name" in config_dict:
        config.name = config_dict["name"]
    if "seed" in config_dict:
        config.seed = config_dict["seed"]
    if "output_base_dir" in config_dict:
        config.output_base_dir = config_dict["output_base_dir"]
    if "write_metrics_dir" in config_dict:
        config.write_metrics_dir = config_dict["write_metrics_dir"]
    if "construction" in config_dict:
        config.construction = config_dict["construction"]
    if "materials" in config_dict:
        config.materials = MaterialsConfig(**config_dict["materials"])
    if "floor_plan" in config_dict:
        config.floor_plan = FloorPlanConfig(**config_dict["floor_plan"])
    if "ahu" in config_dict:
        config.ahu = AHUConfig(**config_dict["ahu"])
    if "weather" in config_dict:
        config.weather = WeatherConfig(**config_dict["weather"])
    if "hot_water_system" in config_dict:
        config.hot_water_system = HotWaterSystemConfig(**config_dict["hot_water_system"])
    if "setpoint_schedule" in config_dict:
        config.setpoint_schedule = SetpointScheduleConfig(
            **config_dict["setpoint_schedule"]
        )
    if "reward" in config_dict:
        config.reward = RewardConfig(**config_dict["reward"])
    if "simulation" in config_dict:
        config.simulation = SimulationConfig(**config_dict["simulation"])

    return config


def _create_floor_plan_generator(fp: FloorPlanConfig):
    """Create a floor plan generator with a fresh import of floor_generator."""

    kwargs = {
        "width": fp.width,
        "height": fp.height,
        "min_room_size": fp.min_room_size,
        "max_room_size": fp.max_room_size,
        "wall_thickness": fp.wall_thickness,
        "door_width": fp.door_width,
        "split_variance": fp.split_variance,
        "building_shape": fp.building_shape,
        "shape_ratio": fp.shape_ratio,
        "composite_coverage": fp.composite_coverage,
        "composite_style": fp.composite_style,
    }
    if fp.num_floors > 1:
        return MultiFloorPlan(num_floors=fp.num_floors, **kwargs)
    return BSPFloorPlan(**kwargs)


# Weather preset directory, smart_buildings/weather.
#
# This walked up one level too many and landed on <repo>/weather, which does not
# exist, so list_weather_presets() returned an empty list and every preset lookup
# by name failed. It went unnoticed because run_scenario_generator carries its own
# correct DEFAULT_WEATHER_DIR and always passes a full path.
#   utils -> smart_control -> smart_buildings
WEATHER_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "weather",
)


def load_weather_config(weather_path_or_name: str) -> WeatherConfig:
    """Load weather configuration from a file or preset name.

    Args:
        weather_path_or_name: Either a path to a YAML file, or a preset name
            (e.g., "fairbanks_1", "phoenix_6").

    Returns:
        WeatherConfig object.

    Raises:
        ValueError: if the preset has no ``weather_csv``. Such a preset drives
            the simulator from a temperature sinusoid between ``high_temp`` and
            ``low_temp`` instead of recorded data. The six legacy presets
            (cold_winter, hot_summer, mild_summer, mild_winter, temperate,
            tropical) are all of this kind and span only a few kelvin, so a
            scenario generated from one is far milder than the real weather it
            appears to name. They are refused rather than generated, because
            the resulting numbers look like climate figures and are not.
    """
    # Check if it's a preset name
    if not weather_path_or_name.endswith(".yaml") and not os.path.exists(
        weather_path_or_name
    ):
        # Try as preset name
        preset_path = os.path.join(WEATHER_DIR, f"{weather_path_or_name}.yaml")
        if os.path.exists(preset_path):
            weather_path_or_name = preset_path
        else:
            raise ValueError(
                f"Weather preset '{weather_path_or_name}' not found. "
                f"Available presets: {list_weather_presets()}"
            )

    with open(weather_path_or_name, "r") as f:
        raw = yaml.safe_load(f)

    if not raw.get("weather_csv"):
        raise ValueError(
            f"Weather preset '{raw.get('name', weather_path_or_name)}' has no "
            f"'weather_csv' key, so it would drive the simulator from a "
            f"synthetic sinusoid rather than recorded weather. Use one of the "
            f"recorded presets: {sorted(list_recorded_weather_presets())}. "
            f"To generate synthetic weather on purpose, construct WeatherConfig "
            f"directly instead of going through a preset."
        )

    return WeatherConfig(
        high_temp=raw.get("high_temp", 305.0),
        low_temp=raw.get("low_temp", 280.0),
        convection_coefficient=raw.get("convection_coefficient", 100.0),
        weather_csv=raw.get("weather_csv", ""),
        start_timestamp=raw.get("start_timestamp", ""),
    )


def list_recorded_weather_presets() -> list[str]:
    """Presets backed by recorded weather, i.e. those carrying a weather_csv."""
    out = []
    for name in list_weather_presets():
        path = os.path.join(WEATHER_DIR, f"{name}.yaml")
        try:
            with open(path, "r") as f:
                if (yaml.safe_load(f) or {}).get("weather_csv"):
                    out.append(name)
        except OSError:
            continue
    return out


def list_weather_presets() -> list[str]:
    """List available weather presets."""
    if not os.path.exists(WEATHER_DIR):
        return []
    return [
        f.replace(".yaml", "") for f in os.listdir(WEATHER_DIR) if f.endswith(".yaml")
    ]


def load_building_config(building_path: str) -> ScenarioConfig:
    """Load building configuration (without weather) from a YAML file.

    Args:
        building_path: Path to the building YAML configuration file.

    Returns:
        ScenarioConfig object (weather will use defaults).
    """
    return load_config(building_path)


def load_scenario_from_parts(
    building_path: str,
    weather_path_or_name: str,
    output_base_dir: str | None = None,
    policy: str | None = None,
) -> ScenarioConfig:
    """Load a scenario from separate building and weather configs.

    Args:
        building_path: Path to the building YAML configuration file.
        weather_path_or_name: Path to weather YAML or preset name (e.g.,
            "phoenix_6"). Must be one of the recorded strips; the legacy
            synthetic presets are refused by load_weather_config.
        output_base_dir: Override for output directory (optional).
        policy: Policy name to include in output path (optional, prevents race conditions).

    Returns:
        ScenarioConfig with building + weather merged.
    """
    # Load building config
    config = load_building_config(building_path)

    # Load and apply weather config
    weather = load_weather_config(weather_path_or_name)
    config.weather = weather

    # Extract weather name for unique output directory
    weather_name = weather_path_or_name
    if weather_path_or_name.endswith(".yaml"):
        weather_name = os.path.basename(weather_path_or_name).replace(".yaml", "")
    elif os.path.sep in weather_path_or_name:
        weather_name = os.path.basename(weather_path_or_name)

    # Include weather (and policy if provided) in output path to avoid race conditions
    if policy:
        config.name = f"{config.name}/{weather_name}/{policy}"
    else:
        config.name = f"{config.name}/{weather_name}"

    # Override output dir if specified
    if output_base_dir:
        config.output_base_dir = output_base_dir

    return config


def _generate_gin_config(
    floor_plan_path: str,
    zone_map_path: str,
    ahu_assignments: list[list[str]],
    config: ScenarioConfig,
) -> str:
    """Generate a complete gin configuration file.

    Args:
        floor_plan_path: Path to the floor plan .npy file.
        zone_map_path: Path to the zone map .npy file.
        ahu_assignments: List of room assignments per AHU.
        config: Full scenario configuration.

    Returns:
        Gin configuration as a string.
    """
    num_ahus = len(ahu_assignments)
    ahu = config.ahu
    construction_name = config.construction
    if config.materials is not None:
        materials = config.materials
        construction_name = f"{construction_name} (overridden)"
    elif construction_name in CONSTRUCTIONS:
        materials = CONSTRUCTIONS[construction_name]
    else:
        raise ValueError(
            f"unknown construction {construction_name!r}; "
            f"choose one of {sorted(CONSTRUCTIONS)} or supply `materials`"
        )
    weather = config.weather
    hws = config.hot_water_system
    schedule = config.setpoint_schedule
    reward = config.reward
    sim = config.simulation
    if config.write_metrics_dir:
        metrics_path = os.path.join(os.path.dirname(floor_plan_path), "metrics")
        os.makedirs(metrics_path, exist_ok=True)
        metrics_path_gin = f'"{metrics_path}"'
    else:
        metrics_path_gin = "None"

    # Room lists for each AHU
    room_lists = []
    for i, rooms in enumerate(ahu_assignments, 1):
        room_strs = ", ".join([f"'{r}'" for r in rooms])
        room_lists.append(f"ahu_{i}_rooms = [{room_strs}]")
    room_lists_str = "\n".join(room_lists)

    # AHU definitions
    ahu_defs = []
    for i in range(1, num_ahus + 1):
        ahu_defs.append(
            f"""ahu_{i}/AirHandler:
  recirculation = %air_handler_recirculation_ratio
  supply_air_temperature_setpoint = %supply_air_temperature_setpoint
  fan_static_pressure = %fan_static_pressure
  fan_efficiency = %fan_efficiency
  max_air_flow_rate = {ahu.sized_max_air_flow_rate}
  min_supply_air_static_pressure = %min_static_pressure
  max_recirculation = %max_recirculation
  sim_weather_controller = %weather_controller
  device_id = 'ahu_{i}'"""
        )
    ahu_defs_str = "\n\n".join(ahu_defs)

    # AHU mappings
    ahu_mappings = ",\n    ".join(
        [f"@ahu_{i}/AirHandler(): %ahu_{i}_rooms" for i in range(1, num_ahus + 1)]
    )

    # Default actions
    ahu_default_actions = []
    for i in range(1, num_ahus + 1):
        ahu_default_actions.extend(
            [
                f"'ahs_ahu_{i}_supply_air_temperature_setpoint': 293.0",
                f"'ahs_ahu_{i}_supply_air_static_pressure_setpoint': 2000.0",
                f"'ahs_ahu_{i}_supervisor_run_command': 1.0",
                f"'ahs_ahu_{i}_recirculation_percentage_command': {ahu.recirculation_ratio}",
            ]
        )
    ahu_default_actions_str = ",\n        ".join(ahu_default_actions)

    # Action normalizers
    ahu_normalizers = []
    for i in range(1, num_ahus + 1):
        ahu_normalizers.extend(
            [
                f"'ahu_{i}_supply_air_temperature_setpoint': @supply_air_temperature_setpoint/set_action_normalization_constants()",
                f"'ahu_{i}_supply_air_static_pressure_setpoint': @static_pressure_setpoint/set_action_normalization_constants()",
                f"'ahu_{i}_supervisor_run_command': @run_command/set_action_normalization_constants()",
                f"'ahu_{i}_recirculation_percentage_command': @percentage_command/set_action_normalization_constants()",
            ]
        )
    ahu_normalizers_str = ",\n        ".join(ahu_normalizers)

    # Device action tuples
    device_action_tuples = []
    for i in range(1, num_ahus + 1):
        device_action_tuples.extend(
            [
                f"('ahs', 'ahu_{i}_supervisor_run_command')",
                f"('ahs', 'ahu_{i}_supply_air_temperature_setpoint')",
                f"('ahs', 'ahu_{i}_supply_air_static_pressure_setpoint')",
                f"('ahs', 'ahu_{i}_recirculation_percentage_command')",
            ]
        )
    device_action_tuples.extend(
        [
            "('hws', 'supervisor_run_command')",
            "('hws', 'supply_water_temperature_setpoint')",
            "('hws', 'differential_pressure_setpoint')",
        ]
    )
    device_action_tuples_str = ",\n        ".join(device_action_tuples)

    # Two weather sources. The sinusoid is the historical default: a fixed daily
    # curve between low and high, identical every day. A preset that names a CSV
    # switches to ReplayWeatherController, which interpolates real hourly
    # observations to whatever timestep the simulation runs at, so the scenario
    # gets fronts, cold snaps and heat waves instead of the same day repeated.
    if weather.weather_csv:
        csv_path = weather.weather_csv
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(WEATHER_DIR, csv_path)
        csv_path = os.path.abspath(csv_path)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"weather_csv not found: {csv_path}")
        # Emit a repo-relative path. An absolute one bakes this machine's home
        # directory into all 60 configs and breaks the moment they are copied to
        # a cluster. Everything is already required to run from the repo root.
        repo_root = os.path.dirname(os.path.dirname(WEATHER_DIR))
        csv_path = os.path.relpath(csv_path, repo_root)
        weather_gin = f"""convection_coefficient = {weather.convection_coefficient}
weather_csv_path = '{csv_path}'

sim/ReplayWeatherController:
  local_weather_path = %weather_csv_path
  convection_coefficient = %convection_coefficient

weather_controller = @sim/ReplayWeatherController()"""
    else:
        weather_gin = f"""convection_coefficient = {weather.convection_coefficient}
ambient_high_temp = {weather.high_temp}
ambient_low_temp = {weather.low_temp}

sim/WeatherController:
  default_low_temp = %ambient_low_temp
  default_high_temp = %ambient_high_temp
  convection_coefficient = %convection_coefficient

weather_controller = @sim/WeatherController()"""

    # A replay preset pins the month, because the file spans a whole year and the
    # episode has to land in the season the preset is named after.
    episode_start = weather.start_timestamp or sim.start_timestamp

    gin_config = f"""###############################################################
# Scenario: {config.name}
# Generated by scenario_generator.py
###############################################################

##########################
### WEATHER CONTROLLER
##########################

{weather_gin}

##########################
### BUILDING
##########################

initial_temp = 294.0
control_volume_cm = {CONTROL_VOLUME_CM}
floor_height_cm = {FLOOR_HEIGHT_CM}

floor_plan_filepath = "{floor_plan_path}"
zone_map_filepath = "{zone_map_path}"

# Material properties. Only density * heat_capacity reaches the solver, as a
# volumetric heat capacity in J/m3K; splitting it in two is for readability.
# The conductivities are effective values for a control volume, not handbook
# values for a bulk material.

# One control volume stands in for a whole facade, so its conductivity is an
# area-weighted blend of insulated opaque wall and glazing. Heat leaving the
# building crosses half that CV, then the surface film, so
# 1/U = 0.1 / k + 0.299.
#
# Construction: {construction_name}
#   effective facade U   {materials.facade_u_value:.2f} W/m2K
#   envelope capacity    {materials.envelope_volumetric_capacity:.2f} MJ/m3K
#
# These vary per building on purpose. They set UA and the fabric capacity, so
# together they decide the free-float equilibrium and the thermal time
# constant -- which is what makes one building's weather hard and another's
# easy. Held identical, every building settles to the same equilibrium above
# ambient and differs only in shape.
exterior_cv_conductivity = {materials.exterior_cv_conductivity}
exterior_cv_density = {materials.exterior_cv_density}
exterior_cv_heat_capacity = {materials.exterior_cv_heat_capacity}

# Interior partitions: gypsum board on a stud cavity, smeared over the CV.
# 0.5 W/mK puts a partition near 2.5 W/m2K, which is what keeps neighbouring
# zones thermally distinct.
interior_wall_cv_conductivity = {materials.interior_wall_cv_conductivity}
interior_wall_cv_density = {materials.interior_wall_cv_density}
interior_wall_cv_heat_capacity = {materials.interior_wall_cv_heat_capacity}

# Occupied space: room air at its own density and heat capacity. The
# conductivity is deliberately far above still air's 0.03, because room air is
# mixed rather than stagnant and a high effective k is what holds a room close
# to isothermal. The furnishings are modelled separately, as the interior
# thermal mass node below, rather than being smeared into this density.
interior_cv_conductivity = {materials.interior_cv_conductivity}
interior_cv_density = {materials.interior_cv_density}
interior_cv_heat_capacity = {materials.interior_cv_heat_capacity}

# Interior thermal mass: the floor slab and the furnishings, as a second node
# on each occupied CV, coupled to the room air across the CV's height. Only the
# top few centimetres of a slab take part in a daily cycle, so the effective
# volumetric capacity is far below concrete's: rho * c * z comes to 69 kJ per
# m2 of floor and K, on top of the 165 the partitions, facade and air already
# hold. k / z sets a 5 W/m2K coupling, which leaves the slab about four hours
# behind the room. Storage only, and deliberately: the node touches nothing but
# its own CV, so no value here can move the building's UA.
interior_mass_cv_conductivity = {materials.interior_mass_cv_conductivity}
interior_mass_cv_density = {materials.interior_mass_cv_density}
interior_mass_cv_heat_capacity = {materials.interior_mass_cv_heat_capacity}

inside_air_properties/MaterialProperties:
  conductivity = %interior_cv_conductivity
  heat_capacity = %interior_cv_heat_capacity
  density = %interior_cv_density

inside_wall_properties/MaterialProperties:
  conductivity = %interior_wall_cv_conductivity
  heat_capacity = %interior_wall_cv_heat_capacity
  density = %interior_wall_cv_density

building_exterior_properties/MaterialProperties:
  conductivity = %exterior_cv_conductivity
  heat_capacity = %exterior_cv_heat_capacity
  density = %exterior_cv_density

interior_mass_properties/MaterialProperties:
  conductivity = %interior_mass_cv_conductivity
  heat_capacity = %interior_mass_cv_heat_capacity
  density = %interior_mass_cv_density

# shuffle parameters
# p: likelihood of any air volume being shuffled per time step
StochasticConvectionSimulator.p = 0.1
# max jump distance in grid units (a radius, no longer a squared value)
StochasticConvectionSimulator.distance = 2
StochasticConvectionSimulator.seed = 5

sim/FloorPlanBasedBuilding:
  cv_size_cm = %control_volume_cm
  floor_height_cm = %floor_height_cm
  initial_temp = %initial_temp
  inside_air_properties = @inside_air_properties/MaterialProperties()
  inside_wall_properties = @inside_wall_properties/MaterialProperties()
  building_exterior_properties = @building_exterior_properties/MaterialProperties()
  interior_mass_properties = @interior_mass_properties/MaterialProperties()
  floor_plan_filepath = %floor_plan_filepath
  zone_map_filepath = %zone_map_filepath
  convection_simulator = @StochasticConvectionSimulator()

##########################
### SETPOINT SCHEDULE
##########################

morning_start_hour = {schedule.morning_start_hour}
evening_start_hour = {schedule.evening_start_hour}
heating_setpoint_day = {schedule.heating_setpoint_day}
cooling_setpoint_day = {schedule.cooling_setpoint_day}
heating_setpoint_night = {schedule.heating_setpoint_night}
cooling_setpoint_night = {schedule.cooling_setpoint_night}
time_zone = "{schedule.time_zone}"

hvac/SetpointSchedule:
  morning_start_hour = %morning_start_hour
  evening_start_hour = %evening_start_hour
  comfort_temp_window = (%heating_setpoint_day, %cooling_setpoint_day)
  eco_temp_window = (%heating_setpoint_night, %cooling_setpoint_night)
  time_zone = %time_zone

##########################
### HVAC SYSTEM
##########################

water_pump_differential_head = {hws.pump_differential_head}
water_pump_efficiency = {hws.pump_efficiency}
supply_water_temperature_setpoint = {hws.supply_water_temperature_setpoint}
boiler_heating_rate = {hws.boiler_heating_rate}
boiler_cooling_rate = {hws.boiler_cooling_rate}
fan_static_pressure = {ahu.fan_static_pressure}
fan_efficiency = {ahu.fan_efficiency}
supply_air_temperature_setpoint = {ahu.supply_air_temperature_setpoint}
air_handler_recirculation_ratio = {ahu.recirculation_ratio}
vav_max_air_flowrate = {VAV_MAX_AIR_FLOWRATE}
vav_reheat_water_flowrate = 0.108

# The ventilation floors. A zone that is occupied has to keep getting outside
# air, so there are three of them and all three have to hold at once: the VAV
# damper cannot close, the fan underneath it cannot stop pushing (zone flow goes
# as sqrt of static pressure, so a damper floor alone is worth nothing), and the
# air handler cannot recirculate everything. Turning an air handler off is still
# allowed and still stops its ventilation; these bound what a running unit may
# be told to do, not whether it runs.
min_damper_setting = 0.1
min_static_pressure = 200.0
max_recirculation = 0.7

{room_lists_str}

{ahu_defs_str}

hvac/AirHandlerSystem:
  ahus = {{
    {ahu_mappings}
  }}
  device_id = 'ahs'

hvac/WaterPump:
  water_pump_differential_head = %water_pump_differential_head
  water_pump_efficiency = %water_pump_efficiency

hvac/Boiler:
  supply_water_temperature_setpoint = %supply_water_temperature_setpoint
  heating_rate = %boiler_heating_rate
  cooling_rate = %boiler_cooling_rate

hvac/HotWaterSystem:
  pump = @hvac/WaterPump()
  heat_source = @hvac/Boiler()
  device_id = 'hws'

sim/FloorPlanBasedHvac:
  air_handler = @hvac/AirHandlerSystem()
  hot_water_system = @hvac/HotWaterSystem()
  schedule = @hvac/SetpointSchedule()
  vav_max_air_flow_rate = %vav_max_air_flowrate
  vav_reheat_max_water_flow_factor = %vav_reheat_water_flowrate
  vav_min_damper_setting = %min_damper_setting

##########################
### SIMULATOR
##########################

time_step_sec = {sim.time_step_sec}
internal_gain_w_per_m2 = {sim.internal_gain_w_per_m2}
internal_gain_standby_fraction = {sim.internal_gain_standby_fraction}
start_timestamp = '{episode_start}'

sim/to_timestamp.date_str = %start_timestamp

sim_building/SimulatorFlexibleGeometries:
  building = @sim/FloorPlanBasedBuilding()
  hvac = @sim/FloorPlanBasedHvac()
  weather_controller = %weather_controller
  time_step_sec = %time_step_sec
  internal_gain_w_per_m2 = %internal_gain_w_per_m2
  internal_gain_standby_fraction = %internal_gain_standby_fraction
  start_timestamp = @sim/to_timestamp()

##########################
### OCCUPANCY
##########################

work_occupancy = {sim.occupants_per_zone}

earliest_expected_arrival_hour = 6
latest_expected_arrival_hour = 12
earliest_expected_departure_hour = 13
latest_expected_departure_hour = 19


randomized_occupancy/RandomizedArrivalDepartureOccupancy:
  zone_assignment = %work_occupancy
  earliest_expected_arrival_hour = %earliest_expected_arrival_hour
  latest_expected_arrival_hour = %latest_expected_arrival_hour
  earliest_expected_departure_hour = %earliest_expected_departure_hour
  latest_expected_departure_hour = %latest_expected_departure_hour
  time_step_sec = %time_step_sec
  time_zone = %time_zone

SimulatorBuilding.simulator = @sim_building/SimulatorFlexibleGeometries()
SimulatorBuilding.occupancy = @randomized_occupancy/RandomizedArrivalDepartureOccupancy()

##########################
### REWARD FUNCTION
##########################

max_productivity_personhour_usd = {reward.max_productivity_personhour_usd}
min_productivity_personhour_usd = {reward.min_productivity_personhour_usd}
productivity_deviation_ref = {reward.productivity_deviation_ref}
max_electricity_rate = {reward.c_electricity_per_ahu * num_ahus}  # {reward.c_electricity_per_ahu:,.0f} W/AHU x {num_ahus} AHUs
max_natural_gas_rate = {reward.c_natural_gas_per_ahu * num_ahus}  # {reward.c_natural_gas_per_ahu:,.0f} W/AHU x {num_ahus} AHUs
productivity_weight = {reward.productivity_weight}
energy_cost_weight = {reward.energy_cost_weight}
carbon_emission_weight = {reward.carbon_emission_weight}

SetpointEnergyCarbonRegretFunction.max_productivity_personhour_usd = %max_productivity_personhour_usd
SetpointEnergyCarbonRegretFunction.min_productivity_personhour_usd = %min_productivity_personhour_usd
SetpointEnergyCarbonRegretFunction.max_electricity_rate = %max_electricity_rate
SetpointEnergyCarbonRegretFunction.max_natural_gas_rate = %max_natural_gas_rate
SetpointEnergyCarbonRegretFunction.productivity_deviation_ref = %productivity_deviation_ref
SetpointEnergyCarbonRegretFunction.electricity_energy_cost = @ElectricityEnergyCost()
SetpointEnergyCarbonRegretFunction.natural_gas_energy_cost = @NaturalGasEnergyCost()
SetpointEnergyCarbonRegretFunction.productivity_weight = %productivity_weight
SetpointEnergyCarbonRegretFunction.energy_cost_weight = %energy_cost_weight
SetpointEnergyCarbonRegretFunction.carbon_emission_weight = %carbon_emission_weight

##########################
### ACTION NORMALIZATION
##########################

supply_water_bounded_action_normalizer/set_action_normalization_constants.min_normalized_value = -1.
supply_water_bounded_action_normalizer/set_action_normalization_constants.max_normalized_value = 1.0
supply_water_bounded_action_normalizer/set_action_normalization_constants.min_native_value = 310
supply_water_bounded_action_normalizer/set_action_normalization_constants.max_native_value = 350.0

supply_air_temperature_setpoint/set_action_normalization_constants.min_normalized_value = -1.
supply_air_temperature_setpoint/set_action_normalization_constants.max_normalized_value = 1.
supply_air_temperature_setpoint/set_action_normalization_constants.min_native_value = 285
supply_air_temperature_setpoint/set_action_normalization_constants.max_native_value = 305.0

differential_pressure_setpoint/set_action_normalization_constants.min_normalized_value = -1.
differential_pressure_setpoint/set_action_normalization_constants.max_normalized_value = 1.
differential_pressure_setpoint/set_action_normalization_constants.min_native_value = 0
differential_pressure_setpoint/set_action_normalization_constants.max_native_value = 20.0

static_pressure_setpoint/set_action_normalization_constants.min_normalized_value = -1.
static_pressure_setpoint/set_action_normalization_constants.max_normalized_value = 1.
static_pressure_setpoint/set_action_normalization_constants.min_native_value = 0
static_pressure_setpoint/set_action_normalization_constants.max_native_value = 2000.0

run_command/set_action_normalization_constants.min_normalized_value = -1.
run_command/set_action_normalization_constants.max_normalized_value = 1.
run_command/set_action_normalization_constants.min_native_value = 0.0
run_command/set_action_normalization_constants.max_native_value = 1.0

# Shared by every 0-to-1 position: VAV damper, VAV reheat valve, AHU
# recirculation. The full native range is offered; the simulator applies the
# ventilation floors, so a command below one of them is honoured as far as it
# legally can be rather than rejected.
percentage_command/set_action_normalization_constants.min_normalized_value = -1.
percentage_command/set_action_normalization_constants.max_normalized_value = 1.
percentage_command/set_action_normalization_constants.min_native_value = 0.0
percentage_command/set_action_normalization_constants.max_native_value = 1.0

action_normalizer_map = {{
        'supply_water_temperature_setpoint': @supply_water_bounded_action_normalizer/set_action_normalization_constants(),
        'differential_pressure_setpoint': @differential_pressure_setpoint/set_action_normalization_constants(),
        {ahu_normalizers_str},
        'supervisor_run_command': @run_command/set_action_normalization_constants(),
        'supply_air_damper_percentage_command': @percentage_command/set_action_normalization_constants(),
        'reheat_valve_percentage_command': @percentage_command/set_action_normalization_constants(),
    }}

ActionConfig:
    action_normalizers = %action_normalizer_map

# The two zone entries are keyed by setpoint name rather than by field id, which
# covers every VAV in the building with one line. The VAVs are named from the
# floor plan at runtime, so there is no list of them to write here.
default_actions = {{
        'hws_supply_water_temperature_setpoint': 340.0,
        'hws_differential_pressure_setpoint': 20.0,
        {ahu_default_actions_str},
        'hws_supervisor_run_command': 1.0,
        'supply_air_damper_percentage_command': 1.0,
        'reheat_valve_percentage_command': 0.0,
    }}

{OBSERVATION_NORMALIZER_CONFIG}

##########################
### ENVIRONMENT
##########################

discount_factor = {sim.discount_factor}
num_days_in_episode = {sim.num_days_in_episode}
metrics_reporting_interval = 10
label = '{config.name}'
metrics_path = {metrics_path_gin}
num_hod_features = 12
num_dow_features = 12

Environment.building = @SimulatorBuilding()
Environment.reward_function = @SetpointEnergyCarbonRegretFunction()
Environment.observation_normalizer = @StandardScoreObservationNormalizer()
Environment.action_config = @ActionConfig()
Environment.metrics_reporting_interval = %metrics_reporting_interval
Environment.metrics_path = %metrics_path
Environment.discount_factor = %discount_factor
Environment.label = %label
Environment.num_days_in_episode = %num_days_in_episode
Environment.default_actions = %default_actions
Environment.num_hod_features = %num_hod_features
Environment.num_dow_features = %num_dow_features

HybridActionEnvironment.building = @SimulatorBuilding()
HybridActionEnvironment.reward_function = @SetpointEnergyCarbonRegretFunction()
HybridActionEnvironment.observation_normalizer = @StandardScoreObservationNormalizer()
HybridActionEnvironment.action_config = @ActionConfig()
HybridActionEnvironment.metrics_reporting_interval = %metrics_reporting_interval
HybridActionEnvironment.metrics_path = %metrics_path
HybridActionEnvironment.discount_factor = %discount_factor
HybridActionEnvironment.label = %label
HybridActionEnvironment.num_days_in_episode = %num_days_in_episode
HybridActionEnvironment.default_actions = %default_actions
HybridActionEnvironment.num_hod_features = %num_hod_features
HybridActionEnvironment.num_dow_features = %num_dow_features
HybridActionEnvironment.device_action_tuples = [
        {device_action_tuples_str},
    ]
HybridActionEnvironment.zone_action_fields = [
        'supply_air_damper_percentage_command',
        'reheat_valve_percentage_command',
    ]
"""
    return gin_config


def generate_scenario(config_path: str) -> dict[str, Any]:
    """Generate a complete simulation scenario from a YAML config file.

    Creates a directory with the scenario name containing:
    - floor_plan.npy: Floor plan for the simulator
    - zone_map.npy: Zone map for room detection
    - config.gin: Complete gin configuration
    - config.yaml: Copy of the input config

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary with:
            - name: Scenario name
            - output_dir: Path to output directory
            - floor_plan_path: Path to floor plan file
            - zone_map_path: Path to zone map file
            - gin_config_path: Path to gin config file
            - num_rooms: Number of rooms generated
            - num_ahus: Number of AHUs
            - ahu_assignments: Room assignments per AHU
    """
    # Load configuration
    config = load_config(config_path)

    # Create output directory
    output_dir = os.path.join(config.output_base_dir, config.name)
    os.makedirs(output_dir, exist_ok=True)

    # Use lock to prevent race conditions with seeded random number generators.
    # Without this, concurrent threads can interleave random calls, causing
    # different floor plans to be generated even with the same seed.
    with _floor_plan_generation_lock:
        # Set random seed if specified (must be inside lock)
        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)

        generator = _create_floor_plan_generator(config.floor_plan)
        generator.generate()

        # Reject a sliver before anything is written. The BSP split honours
        # min_room_size, but the composite shape mask is applied afterwards and
        # clips leaves that straddle its boundary, so the realised minimum is
        # not settable from the floor plan parameters -- building_train_5 and
        # building_train_7 share min_room_size 25 and composite_coverage 0.67
        # and come out at 9.36 m2 and 50.88 m2. A zone that small diverges in
        # simulation, because occupant heat is assigned per zone while the HVAC
        # authority acting on it scales with floor area. The building refuses to
        # construct one; catching it here means a bad seed costs a message
        # instead of a scenario tree that only fails when it is loaded.
        _check_generated_zone_areas(
            generator, CONTROL_VOLUME_CM, config.seed
        )

        # Save floor plan and zone map (still inside lock to prevent partial writes)
        floor_plan_path = os.path.join(output_dir, "floor_plan.npy")
        zone_map_path = os.path.join(output_dir, "zone_map.npy")
        generator.save_for_simulator(floor_plan_path, zone_map_path)

        # Get room count and assign to AHUs (uses seeded random, so keep in lock)
        num_rooms = generator.get_num_rooms()
        # num_ahus is a floor. The cap on zones per handler decides the rest,
        # so a floor plan that grew more rooms gets more handlers instead of
        # quietly overloading the ones it has.
        n_ahus = max(
            config.ahu.num_ahus,
            required_ahus(num_rooms, config.ahu.max_zones_per_ahu),
        )
        ahu_assignments = assign_rooms_to_ahus(
            num_rooms,
            n_ahus,
            seed=config.seed,
            max_rooms_per_ahu=config.ahu.max_zones_per_ahu,
        )

    # Generate and save gin config
    gin_config = _generate_gin_config(
        floor_plan_path,
        zone_map_path,
        ahu_assignments,
        config,
    )
    gin_config_path = os.path.join(output_dir, "config.gin")
    with open(gin_config_path, "w") as f:
        f.write(gin_config)

    # Copy input config to output directory, unless it is already that file. A
    # scenario regenerated in place is driven by the config.yaml already sitting
    # in its own output directory, and copying a file onto itself raises
    # SameFileError. Guarding it here keeps in-place regeneration working;
    # without the guard the run aborts at the very last step, after config.gin
    # and both .npy files have already been written correctly.
    import shutil

    config_copy_path = os.path.join(output_dir, "config.yaml")
    if os.path.realpath(config_path) != os.path.realpath(config_copy_path):
        shutil.copy2(config_path, config_copy_path)

    return {
        "name": config.name,
        "output_dir": output_dir,
        "floor_plan_path": floor_plan_path,
        "zone_map_path": zone_map_path,
        "gin_config_path": gin_config_path,
        "num_rooms": num_rooms,
        "num_ahus": len(ahu_assignments),
        "ahu_assignments": ahu_assignments,
        "bsp": generator,
        "config": config,
    }


def generate_scenario_from_config(config: ScenarioConfig) -> dict[str, Any]:
    """Generate a complete simulation scenario from a ScenarioConfig object.

    Same as generate_scenario but accepts a ScenarioConfig object directly.
    Useful when building/weather configs are loaded separately.

    Args:
        config: ScenarioConfig object with all parameters.

    Returns:
        Same as generate_scenario().
    """
    # Create output directory
    output_dir = os.path.join(config.output_base_dir, config.name)
    os.makedirs(output_dir, exist_ok=True)

    # Use lock to prevent race conditions with seeded random number generators.
    # Without this, concurrent threads can interleave random calls, causing
    # different floor plans to be generated even with the same seed.
    with _floor_plan_generation_lock:
        # Set random seed if specified (must be inside lock)
        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)

        generator = _create_floor_plan_generator(config.floor_plan)
        generator.generate()

        # Reject a sliver before anything is written. The BSP split honours
        # min_room_size, but the composite shape mask is applied afterwards and
        # clips leaves that straddle its boundary, so the realised minimum is
        # not settable from the floor plan parameters -- building_train_5 and
        # building_train_7 share min_room_size 25 and composite_coverage 0.67
        # and come out at 9.36 m2 and 50.88 m2. A zone that small diverges in
        # simulation, because occupant heat is assigned per zone while the HVAC
        # authority acting on it scales with floor area. The building refuses to
        # construct one; catching it here means a bad seed costs a message
        # instead of a scenario tree that only fails when it is loaded.
        _check_generated_zone_areas(
            generator, CONTROL_VOLUME_CM, config.seed
        )

        # Save floor plan and zone map (still inside lock to prevent partial writes)
        floor_plan_path = os.path.join(output_dir, "floor_plan.npy")
        zone_map_path = os.path.join(output_dir, "zone_map.npy")
        generator.save_for_simulator(floor_plan_path, zone_map_path)

        # Get room count and assign to AHUs (uses seeded random, so keep in lock)
        num_rooms = generator.get_num_rooms()
        # num_ahus is a floor. The cap on zones per handler decides the rest,
        # so a floor plan that grew more rooms gets more handlers instead of
        # quietly overloading the ones it has.
        n_ahus = max(
            config.ahu.num_ahus,
            required_ahus(num_rooms, config.ahu.max_zones_per_ahu),
        )
        ahu_assignments = assign_rooms_to_ahus(
            num_rooms,
            n_ahus,
            seed=config.seed,
            max_rooms_per_ahu=config.ahu.max_zones_per_ahu,
        )

    # Generate and save gin config
    gin_config = _generate_gin_config(
        floor_plan_path,
        zone_map_path,
        ahu_assignments,
        config,
    )
    gin_config_path = os.path.join(output_dir, "config.gin")
    with open(gin_config_path, "w") as f:
        f.write(gin_config)

    # Save merged config to output directory
    import dataclasses

    config_copy_path = os.path.join(output_dir, "config.yaml")
    with open(config_copy_path, "w") as f:
        yaml.dump(dataclasses.asdict(config), f, default_flow_style=False)

    return {
        "name": config.name,
        "output_dir": output_dir,
        "floor_plan_path": floor_plan_path,
        "zone_map_path": zone_map_path,
        "gin_config_path": gin_config_path,
        "num_rooms": num_rooms,
        "num_ahus": len(ahu_assignments),
        "ahu_assignments": ahu_assignments,
        "bsp": generator,
        "config": config,
    }


def generate_scenario_from_dict(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Generate a scenario from a configuration dictionary.

    Same as generate_scenario but accepts a dict instead of a file path.
    Useful for programmatic scenario generation.

    Args:
        config_dict: Configuration dictionary matching YAML structure.

    Returns:
        Same as generate_scenario().
    """
    return generate_scenario_from_config(scenario_config_from_dict(config_dict))


def print_scenario_summary(result: dict[str, Any]) -> None:
    """Print a summary of a generated scenario."""
    print(f"\n{'='*60}")
    print(f"Scenario: {result['name']}")
    print(f"{'='*60}")
    print(f"Output directory: {result['output_dir']}")
    print(f"Number of rooms: {result['num_rooms']}")
    print(f"Number of AHUs: {result['num_ahus']}")
    print("\nAHU Assignments:")
    for i, rooms in enumerate(result["ahu_assignments"], 1):
        print(f"  AHU {i}: {len(rooms)} rooms")
    print("\nGenerated files:")
    print(f"  - {result['floor_plan_path']}")
    print(f"  - {result['zone_map_path']}")
    print(f"  - {result['gin_config_path']}")
    print(f"{'='*60}\n")


def get_env(result: dict[str, Any], reset: bool = True):
    """Create an RL environment from a generated scenario.

    Args:
        result: Result dictionary from generate_scenario() or generate_scenario_from_dict().
        reset: Whether to call env.reset() before returning (default: True).

    Returns:
        HybridActionEnvironment configured for the scenario.

    Example:
        result = generate_scenario("scenarios/my_config.yaml")
        env = get_env(result)
        obs, info = env.reset()
        action = {'discrete_action': [1, 1, 0], 'continuous_action': [-1, 1, -1, 1, 1.0, -1]}
        obs, reward, terminated, truncated, info = env.step(action)
    """
    import gin

    # Import all modules that register gin configurables
    # These must be imported BEFORE parsing the gin config
    from smart_buildings.smart_control.environment import hybrid_action_environment

    # Clear any existing gin config and parse the scenario's config
    gin.clear_config()
    gin.parse_config_file(result["gin_config_path"])

    # Create the environment
    env = hybrid_action_environment.HybridActionEnvironment()

    if reset:
        env.reset()

    return env


def get_env_from_config(config_path: str, reset: bool = True):
    """Generate a scenario and create an RL environment in one step.

    Convenience function that combines generate_scenario() and get_env().

    Args:
        config_path: Path to the YAML configuration file.
        reset: Whether to call env.reset() before returning (default: True).

    Returns:
        Tuple of (env, result) where env is the HybridActionEnvironment
        and result is the scenario generation result dict.

    Example:
        env, result = get_env_from_config("scenarios/my_config.yaml")
        print(f"Created env with {result['num_rooms']} rooms")
    """
    result = generate_scenario(config_path)
    env = get_env(result, reset=reset)
    return env, result


def print_action_spec(env) -> dict[str, Any]:
    """Print the action specification for an environment and return an example action.

    Displays the structure of discrete and continuous actions with their
    indices, names, and value ranges. Returns a valid example action dict.

    Args:
        env: HybridActionEnvironment instance.

    Returns:
        Dict with 'discrete_action' and 'continuous_action' lists that can
        be passed directly to env.step().

    Example:
        env = get_env(result)
        example_action = print_action_spec(env)
        obs, reward, terminated, truncated, info = env.step(example_action)
    """
    print("=== ACTION SPEC ===\n")

    print("action = {")
    print(
        f"    'discrete_action': [  # shape: {env.action_spec()['discrete_action'].shape}"
    )
    discrete_actions = []
    for device, action_name in env._device_action_tuples:
        if "run_command" in action_name:
            print(
                f"        # [{len(discrete_actions)}] {device}_{action_name} -> Off=0, On=1"
            )
            discrete_actions.append(1)  # default: On

    print("    ],")
    print(
        f"    'continuous_action': [  # shape: {env.action_spec()['continuous_action'].shape}"
    )
    continuous_actions = []
    for device, action_name in env._device_action_tuples:
        if "run_command" not in action_name:
            key = action_name
            display_key = f"{device}_{action_name}"
            norm = env.action_normalizers.get(key)
            if norm:
                min_v, max_v = norm._min_native_value, norm._max_native_value
                unit = "K" if "temperature" in action_name else "Pa"
                print(
                    f"        # [{len(continuous_actions)}] {display_key} -> [{min_v}, {max_v}] {unit}"
                )
            continuous_actions.append(0.0)  # default: midpoint
    print("    ],")
    print("}")

    example_action = {
        "discrete_action": np.asarray(discrete_actions, dtype=np.int32),
        "continuous_action": np.asarray(continuous_actions, dtype=np.float32),
    }
    print("\n# Example action (all On, all midpoint):")
    print(f"example_action = {example_action}")

    return example_action


class SimulationTracker:
    """Simple tracker for logging and visualizing simulation runs.

    Example:
        env = get_env(result)
        tracker = SimulationTracker(env)

        for i in range(1000):
            tracker.step(action)
            if i % 100 == 0:
                tracker.render()
                tracker.plot()

        # Render specific timestep
        tracker.render(step=500)

        # Get summary
        tracker.summary()
    """

    def __init__(
        self,
        env,
        vmin: float = 280,
        vmax: float = 310,
        capture_snapshots: bool = True,
    ):
        """Initialize tracker.

        Args:
            env: HybridActionEnvironment instance.
            vmin: Min temperature for rendering colormap.
            vmax: Max temperature for rendering colormap.
            capture_snapshots: Retain a copy of the full building temperature
              grid on every step, in `temp_snapshots`. Only `render(step=...)`
              and `create_video()` read it; nothing in the reward, metrics or
              CSV path does. Set False for long headless runs.

              This buffer is large. The grid is float64 and is copied once per
              step, so a 30-day episode at 5-minute steps (8640 steps) retains
              7.2 GB for the smallest building and 27.8 GB for the largest.
              Leaving it on with a 4 GB task limit is what killed the first
              cluster smoke test. Snapshots are downcast to float32 here, which
              halves that and is lossless for rendering, but the buffer still
              grows without bound, so turn it off unless you want a video.
        """
        from smart_buildings.smart_control.utils import building_renderer

        self.env = env
        self.vmin = vmin
        self.vmax = vmax
        self.capture_snapshots = capture_snapshots

        # Create renderer
        building_layout = env.building.simulator.building.floor_plan
        self.renderer = building_renderer.BuildingRenderer(building_layout, 1)

        # Get env info
        self.ahus = env.building.simulator._hvac.air_handler._ahus
        self.vavs = env.building.simulator._hvac._vavs
        self.zone_ids = list(self.vavs.keys())
        self.time_step_sec = env.building.simulator._time_step_sec
        self.num_ahus = len(self.ahus)

        # Build zone-to-AHU index mapping for coloring
        self.zone_ahu_indices = []
        for zone_id in self.zone_ids:
            vav = self.vavs[zone_id]
            ahu_obj = vav._air_handler
            try:
                ahu_idx = self.ahus.index(ahu_obj)
            except ValueError:
                ahu_idx = 0  # fallback
            self.zone_ahu_indices.append(ahu_idx)

        # Setpoint schedule for temperature range overlay
        self.setpoint_schedule = env.building.simulator._hvac.schedule

        # Reward weights
        reward_fn = env.reward_function
        self.productivity_weight = reward_fn._productivity_weight
        self.energy_weight = reward_fn._energy_cost_weight
        self.carbon_weight = reward_fn._carbon_emission_weight

        # Logs
        self.rewards = []
        self.occupancy = []
        self.timestamps = []
        self.temp_snapshots = []  # For rendering past states
        self.zone_temps = []
        self.ahu_log = []
        self.reward_components = []

        self._step_count = 0

    def reset(self):
        """Reset tracker state and environment."""
        self.last_obs, _ = self.env.reset()
        self.rewards = []
        self.occupancy = []
        self.timestamps = []
        self.temp_snapshots = []
        self.zone_temps = []
        self.ahu_log = []
        self.reward_components = []
        self._step_count = 0

    def step(self, action: dict[str, Any]):
        """Step environment and log all data.

        Args:
            action: Action dict with 'discrete_action' and 'continuous_action'.

        Returns:
            The gymnasium step tuple
            (observation, reward, terminated, truncated, info).
        """
        import pandas as pd

        observation, reward, terminated, truncated, info = self.env.step(action)
        reward = float(reward)
        self.last_obs = observation
        timestamp = self.env.building.current_timestamp

        # Store temp snapshot for rendering. Skipped unless explicitly wanted:
        # this is a full-grid copy per step and is the largest allocation in
        # the loop by orders of magnitude. See capture_snapshots in __init__.
        if self.capture_snapshots:
            self.temp_snapshots.append(
                self.env.building.simulator.building.temp.astype(
                    np.float32, copy=True
                )
            )

        # Basic logs
        self.rewards.append(reward)
        self.timestamps.append(timestamp)

        # Reward components
        _, reward_response = self.env.get_reward_info_and_response()
        self.reward_components.append(
            {
                "timestamp": timestamp,
                "agent_reward": reward_response.agent_reward_value,
                "productivity_reward": reward_response.productivity_reward,
                "electricity_energy_cost": reward_response.electricity_energy_cost,
                "natural_gas_energy_cost": reward_response.natural_gas_energy_cost,
                "carbon_emitted": reward_response.carbon_emitted,
                "normalized_productivity_regret": reward_response.normalized_productivity_regret,
                "normalized_energy_cost": reward_response.normalized_energy_cost,
                "normalized_carbon_emission": reward_response.normalized_carbon_emission,
                "total_occupancy": reward_response.total_occupancy,
            }
        )

        # Reuse the reward's room occupancy. self.zone_ids contains VAV IDs;
        # querying occupancy with those creates fictitious populations and
        # consumes random draws, changing subsequent simulation trajectories.
        self.occupancy.append(float(reward_response.total_occupancy))

        # Zone temps
        temps = [vav.zone_air_temperature for vav in self.vavs.values()]
        self.zone_temps.append(temps)

        # AHU data
        avg_temps_dict = self.env.building.simulator.building.get_zone_average_temps()
        recirculation_temp = sum(avg_temps_dict.values()) / len(avg_temps_dict)
        outside_air = self.ahus[0].outside_air_temperature_sensor

        self.ahu_log.append(
            {
                "timestamp": timestamp,
                "outside_air": outside_air,
                "recirculation_temp": recirculation_temp,
                "ahu_supply_temps": [
                    ahu.get_supply_air_temp(recirculation_temp, outside_air)
                    for ahu in self.ahus
                ],
                "ahu_run_cmds": [int(ahu.run_command) for ahu in self.ahus],
                "ahu_flow_rates": [ahu.air_flow_rate for ahu in self.ahus],
            }
        )

        self._step_count += 1
        return observation, reward, terminated, truncated, info

    def render(self, step: int | None = None):
        """Render building temperature visualization.

        Args:
            step: Step index to render. If None, renders current state.

        Returns:
            PIL Image.
        """
        if step is not None:
            if not self.capture_snapshots:
                raise ValueError(
                    "Cannot render a past step: this tracker was built with "
                    "capture_snapshots=False, so no history was kept. Call "
                    "render() with no argument for the current state, or "
                    "rebuild with capture_snapshots=True."
                )
            if step < 0 or step >= len(self.temp_snapshots):
                raise ValueError(
                    f"Step {step} out of range [0, {len(self.temp_snapshots)-1}]"
                )
            temps = self.temp_snapshots[step]
        else:
            temps = self.env.building.simulator.building.temp

        image = self.renderer.render(
            temps, cmap="bwr", vmin=self.vmin, vmax=self.vmax
        ).convert("RGB")

        # IPython is a notebook-only dependency and is not in the pinned
        # install set, so display is best-effort. The image is returned either
        # way, which is what headless callers want.
        try:
            from IPython.display import display

            display(image)
        except ImportError:
            pass

        return image

    def create_video(
        self,
        output_path: str = "simulation.mp4",
        fps: int = 10,
        show_progress: bool = True,
    ) -> str:
        """Create a video from temperature snapshots.

        Args:
            output_path: Output file path. Supports .mp4, .gif, .avi.
            fps: Frames per second.
            show_progress: Whether to show progress bar.

        Returns:
            Path to the created video file.
        """
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont

        if not self.temp_snapshots:
            if not self.capture_snapshots:
                raise ValueError(
                    "No snapshots: this tracker was built with "
                    "capture_snapshots=False. Rebuild it with "
                    "SimulationTracker(env, capture_snapshots=True) and re-run "
                    "the simulation to record frames."
                )
            raise ValueError("No snapshots to create video from. Run simulation first.")

        # Determine output format
        ext = output_path.lower().split(".")[-1]
        if ext not in ("mp4", "gif", "avi"):
            raise ValueError(f"Unsupported format: {ext}. Use mp4, gif, or avi.")

        frames = []
        n_frames = len(self.temp_snapshots)

        iterator = range(n_frames)
        if show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, desc="Rendering frames")
            except ImportError:
                pass

        for i in iterator:
            temps = self.temp_snapshots[i]
            frame = self.renderer.render(
                temps, cmap="bwr", vmin=self.vmin, vmax=self.vmax
            ).convert("RGB")

            # Add timestamp overlay if available
            if i < len(self.timestamps):
                draw = ImageDraw.Draw(frame)
                ts_str = self.timestamps[i].strftime("%Y-%m-%d %H:%M")
                # Try to use a monospace font, fall back to default
                try:
                    font = ImageFont.truetype(
                        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 16
                    )
                except OSError:
                    font = ImageFont.load_default()
                draw.rectangle([5, 5, 180, 25], fill="white")
                draw.text((10, 7), ts_str, fill="black", font=font)

            frames.append(np.array(frame))

        # Write video
        if ext == "gif":
            # Use PIL for GIF
            pil_frames = [Image.fromarray(f) for f in frames]
            pil_frames[0].save(
                output_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=int(1000 / fps),
                loop=0,
            )
        else:
            # Use imageio for mp4/avi
            try:
                import imageio
            except ImportError:
                raise ImportError(
                    "imageio required for mp4/avi. Install with: pip install imageio[ffmpeg]"
                )

            writer = imageio.get_writer(output_path, fps=fps)
            for frame in frames:
                writer.append_data(frame)
            writer.close()

        print(f"Video saved to {output_path} ({n_frames} frames, {fps} fps)")
        return output_path

    def plot(self, color_by_ahu: bool = False):
        """Plot all charts (temps, rewards, occupancy, AHU diagnostics, reward components).

        Args:
            color_by_ahu: If True, color zone temps by their AHU assignment.
        """
        self.plot_overview(color_by_ahu=color_by_ahu)
        self.plot_ahu()
        self.plot_rewards()

    def plot_overview(self, color_by_ahu: bool = False):
        """Plot temperature, reward, and occupancy overview.

        Args:
            color_by_ahu: If True, color zone temps by their AHU assignment.
        """
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt

        if not self.timestamps:
            print("No data to plot.")
            return

        mpl_ts = [ts.to_pydatetime() for ts in self.timestamps]
        n_zones = len(self.zone_temps[0]) if self.zone_temps else 0

        fig, (ax1, ax2, ax3) = plt.subplots(
            3,
            1,
            figsize=(12, 10),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1, 1]},
        )

        # Panel 1: Temperatures with setpoint range overlay
        outside_air = [d["outside_air"] for d in self.ahu_log]

        # Get setpoint schedule data for the time range
        setpoint_data = self.setpoint_schedule.get_plot_data(
            self.timestamps[0], self.timestamps[-1]
        )

        # Draw setpoint ranges as shaded regions (behind zone temps)
        for _, row in setpoint_data.iterrows():
            start = row["start_time"].to_pydatetime()
            end = row["end_time"].to_pydatetime()
            heating = row["heating_setpoint"]
            cooling = row["cooling_setpoint"]
            color = (
                "#90EE90" if row["comfort_mode"] else "#ADD8E6"
            )  # green=day, blue=night
            ax1.fill_between(
                [start, end],
                heating,
                cooling,
                alpha=0.3,
                color=color,
                label="_nolegend_",
            )
            ax1.plot(
                [start, end],
                [heating, heating],
                color=color,
                linewidth=1,
                linestyle="--",
            )
            ax1.plot(
                [start, end],
                [cooling, cooling],
                color=color,
                linewidth=1,
                linestyle="--",
            )

        # Add legend entries for setpoint ranges
        ax1.fill_between(
            [], [], [], alpha=0.3, color="#90EE90", label="Comfort Range (Day)"
        )
        ax1.fill_between(
            [], [], [], alpha=0.3, color="#ADD8E6", label="Eco Range (Night)"
        )

        ax1.plot(mpl_ts, outside_air, "b-", linewidth=2, label="Outside Air")

        # AHU colors for color_by_ahu mode
        ahu_colors = [
            "red",
            "blue",
            "green",
            "orange",
            "purple",
            "brown",
            "pink",
            "cyan",
        ]

        if color_by_ahu:
            # Plot zones colored by their AHU assignment
            ahu_plotted = set()
            for i in range(n_zones):
                zone_trace = [t[i] for t in self.zone_temps]
                ahu_idx = self.zone_ahu_indices[i]
                color = ahu_colors[ahu_idx % len(ahu_colors)]
                if ahu_idx not in ahu_plotted:
                    label = f"AHU {ahu_idx + 1} zones"
                    ahu_plotted.add(ahu_idx)
                else:
                    label = "_nolegend_"
                ax1.plot(
                    mpl_ts, zone_trace, color=color, linewidth=1, alpha=0.5, label=label
                )
        else:
            # Single color for all zones
            for i in range(n_zones):
                zone_trace = [t[i] for t in self.zone_temps]
                label = f"Zone Temps ({n_zones} zones)" if i == 0 else "_nolegend_"
                ax1.plot(
                    mpl_ts,
                    zone_trace,
                    color="#FFD700",
                    linewidth=1,
                    alpha=0.5,
                    label=label,
                )

        ax1.set_ylabel("Temperature (K)")
        ax1.set_title("Temperature Overview")
        ax1.legend(loc="upper left")
        ax1.grid(True, linestyle="--", alpha=0.6)

        # Panel 2: Rewards
        ax2.plot(mpl_ts, self.rewards, "purple", linewidth=2)
        ax2.fill_between(mpl_ts, self.rewards, alpha=0.3, color="purple")
        ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax2.set_ylabel("Reward")
        ax2.set_title(f"Reward (cumulative: {sum(self.rewards):.2f})")
        ax2.grid(True, linestyle="--", alpha=0.6)

        # Panel 3: Occupancy
        ax3.plot(mpl_ts, self.occupancy, "orange", linewidth=2)
        ax3.fill_between(mpl_ts, self.occupancy, alpha=0.3, color="orange")
        ax3.set_ylabel("Occupancy")
        ax3.set_xlabel("Time")
        ax3.set_title(f"Occupancy (avg: {sum(self.occupancy)/len(self.occupancy):.1f})")
        ax3.grid(True, linestyle="--", alpha=0.6)

        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()

    def plot_ahu(self):
        """Plot AHU diagnostics."""
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt

        if not self.ahu_log:
            return

        mpl_ts = [d["timestamp"].to_pydatetime() for d in self.ahu_log]
        colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "cyan"]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        # Supply temps
        for i in range(self.num_ahus):
            temps = [d["ahu_supply_temps"][i] for d in self.ahu_log]
            ax1.plot(
                mpl_ts,
                temps,
                color=colors[i % len(colors)],
                linewidth=2,
                label=f"AHU {i+1} ({temps[-1]:.1f}K)",
            )
        ax1.set_ylabel("Supply Air Temp (K)")
        ax1.set_title("AHU Supply Air Temperatures")
        ax1.legend(loc="upper right")
        ax1.grid(True, linestyle="--", alpha=0.6)

        # Flow rates
        for i in range(self.num_ahus):
            flows = [d["ahu_flow_rates"][i] for d in self.ahu_log]
            ax2.plot(
                mpl_ts,
                flows,
                color=colors[i % len(colors)],
                linewidth=2,
                label=f"AHU {i+1} ({flows[-1]:.2f} m³/s)",
            )
        ax2.set_ylabel("Flow Rate (m³/s)")
        ax2.set_xlabel("Time")
        ax2.set_title("AHU Flow Rates")
        ax2.legend(loc="upper right")
        ax2.grid(True, linestyle="--", alpha=0.6)

        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()

    def plot_rewards(self):
        """Plot reward component breakdown."""
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt

        if not self.reward_components:
            return

        mpl_ts = [d["timestamp"].to_pydatetime() for d in self.reward_components]
        total_weight = (
            self.productivity_weight + self.energy_weight + self.carbon_weight
        )

        prod = [
            d["normalized_productivity_regret"]
            * self.productivity_weight
            / total_weight
            for d in self.reward_components
        ]
        energy = [
            -d["normalized_energy_cost"] * self.energy_weight / total_weight
            for d in self.reward_components
        ]
        carbon = [
            -d["normalized_carbon_emission"] * self.carbon_weight / total_weight
            for d in self.reward_components
        ]
        total = [d["agent_reward"] for d in self.reward_components]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Weighted components
        ax1.plot(
            mpl_ts,
            prod,
            "g-",
            linewidth=2,
            label=f"Productivity ({self.productivity_weight:.0%})",
        )
        ax1.plot(
            mpl_ts,
            energy,
            "r-",
            linewidth=2,
            label=f"Energy ({self.energy_weight:.0%})",
        )
        ax1.plot(
            mpl_ts,
            carbon,
            "orange",
            linewidth=2,
            label=f"Carbon ({self.carbon_weight:.0%})",
        )
        ax1.plot(mpl_ts, total, "purple", linewidth=2, linestyle="--", label="Total")
        ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax1.set_ylabel("Reward Contribution")
        ax1.set_title("Weighted Reward Components")
        ax1.legend(loc="upper right")
        ax1.grid(True, linestyle="--", alpha=0.6)

        # Raw costs
        elec = [d["electricity_energy_cost"] for d in self.reward_components]
        gas = [d["natural_gas_energy_cost"] for d in self.reward_components]
        productivity = [d["productivity_reward"] for d in self.reward_components]

        ax2.plot(
            mpl_ts, elec, "r-", linewidth=2, label=f"Electricity (${elec[-1]:.2f})"
        )
        ax2.plot(mpl_ts, gas, "orange", linewidth=2, label=f"Gas (${gas[-1]:.2f})")
        ax2.plot(
            mpl_ts,
            productivity,
            "g-",
            linewidth=2,
            label=f"Productivity (${productivity[-1]:.2f})",
        )
        ax2.set_ylabel("Cost ($)")
        ax2.set_xlabel("Time")
        ax2.set_title("Raw Costs per Step")
        ax2.legend(loc="upper right")
        ax2.grid(True, linestyle="--", alpha=0.6)

        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()

    def summary(self):
        """Print simulation summary."""
        if not self.rewards:
            print("No data yet.")
            return

        print(f"\n{'='*50}")
        print(f"Simulation Summary ({self._step_count} steps)")
        print(f"{'='*50}")
        print(f"Cumulative Reward: {sum(self.rewards):.4f}")
        print(f"Average Reward: {sum(self.rewards)/len(self.rewards):.4f}")
        print(f"Min/Max Reward: {min(self.rewards):.4f} / {max(self.rewards):.4f}")
        print(f"Average Occupancy: {sum(self.occupancy)/len(self.occupancy):.2f}")
        if self.reward_components:
            rc = self.reward_components[-1]
            print(
                f"Last step costs: Elec=${rc['electricity_energy_cost']:.2f}, "
                f"Gas=${rc['natural_gas_energy_cost']:.2f}, "
                f"Carbon={rc['carbon_emitted']:.2f}kg"
            )
        print(f"{'='*50}\n")


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate HVAC simulation scenarios from YAML configs"
    )
    parser.add_argument("config_path", type=str, help="Path to YAML configuration file")
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress output summary"
    )

    args = parser.parse_args()

    result = generate_scenario(args.config_path)

    if not args.quiet:
        print_scenario_summary(result)
