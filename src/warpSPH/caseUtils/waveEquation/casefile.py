"""TOML casefile loading for the wave-equation demo: parses a declarative
description of sources/obstacles/physics into the dataclasses `configurations`
defines (`SimulationConfig`, `WaveCaseConfig`, `WaveShapeSpec`, ...), so one
casefile can describe a family of runs (see `WaveCaseConfig`'s
``randomize*``/``*Range`` fields). `build_configs_from_casefile` is the entry
point; it is the first of the five stages the module docstring of
`sample/waveSystem.py` and `CLEANUP_PLAN.md` describe as the (currently
unwired -- no runner drives all five, and no TOML casefile ships in this repo)
wave-equation pipeline. `argparse_defaults_from_casefile` maps the same TOML
sections onto the CLI flag names a future runner would use; nothing in this
repo currently calls it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from ...configurations import WaveCaseConfig, WaveShapeSpec, SimulationConfig, WaveBoundary, WaveSource
from ...utils import buildDomainDescription, n_h_to_nH
from ...geometry import SamplingScheme
from warpSPHIntegrators import IntegrationSchemeType
from warpSPHCore import GradientScheme, KernelFunctions, LaplacianScheme, SupportScheme

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

__all__ = ['load_casefile', 'argparse_defaults_from_casefile', 'build_configs_from_casefile']


def load_casefile(casefile: str) -> Dict[str, Any]:
    path = Path(casefile)
    if not path.exists():
        raise FileNotFoundError(f"Case file not found: {casefile}")
    with path.open("rb") as f:
        return tomllib.load(f)


def _shape_spec_from_dict(item: Dict[str, Any]) -> WaveShapeSpec:
    return WaveShapeSpec(
        kind=item.get("kind", "sphere"),
        position=tuple(item.get("position", (0.0, 0.0))),
        rotation=float(item.get("rotation", 0.0)),
        params=dict(item.get("params", {})),
    )


def _list_section(data: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    section = data.get(key, [])
    if isinstance(section, dict):
        return list(section.get("items", []))
    if isinstance(section, list):
        return section
    return []


def argparse_defaults_from_casefile(data: Dict[str, Any]) -> Dict[str, Any]:
    core = data.get("core", {})
    case = data.get("case", {})
    noise = data.get("noise", {})
    physics = data.get("physics", {})

    defaults: Dict[str, Any] = {}
    mapping = {
        "nx": "nx",
        "sampling": "sampling",
        "dt": "dt",
        "time_limit": "timeLimit",
        "adaptive_dt": "adaptiveDt",
        "cfl_factor": "cflFactor",
        "n_iter": "nIter",
        "plot_interval": "plotInterval",
        "file_prefix": "filePrefix",
        "verbose": "verbose",
        "export": "export",
        "export_images": "exportImages",
        "export_initial": "exportInitial",
        "figure_dpi": "figureDpi",
        "case_index": "caseIndex",
        "boundary_case_index": "boundaryCaseIndex",
    }
    for src_key, dst_key in mapping.items():
        if src_key in core:
            defaults[dst_key] = core[src_key]

    case_mapping = {
        "domain_box": "domainBox",
        "domain_damping": "domainDamping",
        "smooth_ics": "smoothICs",
        "smooth_iters": "smoothIters",
    }
    for src_key, dst_key in case_mapping.items():
        if src_key in case:
            defaults[dst_key] = case[src_key]

    noise_mapping = {
        "enable": "enableNoise",
        "type": "noiseType",
        "amplitude": "noiseAmplitude",
        "smooth_iter": "noiseSmoothIter",
        "seed": "noiseSeed",
    }
    for src_key, dst_key in noise_mapping.items():
        if src_key in noise:
            defaults[dst_key] = noise[src_key]

    physics_mapping = {
        "default_speed": "defaultSpeed",
        "boundary_speed": "boundarySpeed",
        "obstacle_speeds": "obstacleSpeeds",
        "random_obstacle_speed": "randomObstacleSpeed",
        "obstacle_speed_min": "obstacleSpeedMin",
        "obstacle_speed_max": "obstacleSpeedMax",
        "u_magnitudes": "uMagnitudes",
        "u_random_magnitude": "uRandomMagnitude",
        "u_random_min": "uRandomMin",
        "u_random_max": "uRandomMax",
    }
    for src_key, dst_key in physics_mapping.items():
        if src_key in physics:
            defaults[dst_key] = physics[src_key]

    return defaults


def _parse_enum(enum_cls, value, default):
    if value is None:
        return default
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        for candidate in enum_cls:
            if candidate.name.lower() == value.lower():
                return candidate
    return default


def build_configs_from_casefile(
    casefile: str,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[SimulationConfig, WaveCaseConfig]:
    data = load_casefile(casefile)

    core = data.get("core", {})
    domain_cfg = data.get("domain", {})
    case = data.get("case", {})
    noise = data.get("noise", {})
    physics = data.get("physics", {})

    if device is None:
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    dim = int(domain_cfg.get("dim", 2))
    domain_min = domain_cfg.get("min", [-1.0, -1.0])
    domain_max = domain_cfg.get("max", [1.0, 1.0])
    periodic = bool(domain_cfg.get("periodic", True))
    l = float(domain_max[0] - domain_min[0])

    n_h = int(core.get("n_h", 4))
    sampling_name = str(core.get("sampling", "regular"))
    sampling_scheme = SamplingScheme.regular
    for scheme in SamplingScheme:
        if scheme.name.lower() == sampling_name.lower():
            sampling_scheme = scheme
            break

    simulation_config = SimulationConfig(
        domain=buildDomainDescription(l, dim, periodic, device, dtype),
        dim=dim,
        kernel=_parse_enum(KernelFunctions, core.get("kernel"), KernelFunctions.Wendland4),
        n_h=n_h,
        targetNeighbors=n_h_to_nH(n_h, dim),
        supportMode=_parse_enum(SupportScheme, core.get("support_mode"), SupportScheme.SuperSymmetric),
        gradientMode=_parse_enum(GradientScheme, core.get("gradient_mode"), GradientScheme.Difference),
        laplacianMode=_parse_enum(LaplacianScheme, core.get("laplacian_mode"), LaplacianScheme.Brookshaw),
        integrationScheme=_parse_enum(IntegrationSchemeType, core.get("integration_scheme"), IntegrationSchemeType.rungeKutta4),
        samplingScheme=sampling_scheme,
        device=device,
        dtype=dtype,
    )

    sources: List[WaveSource] = []
    for item in _list_section(data, "sources"):
        sources.append(
            WaveSource(
                shapeSpec=_shape_spec_from_dict(item),
                magnitude=item.get("magnitude"),
                randomizeMagnitude=bool(item.get("randomize_magnitude", False)),
                radiusRange=tuple(item.get("radius_range", (0.05, 0.15))),
            )
        )

    obstacles: List[WaveBoundary] = []
    for item in _list_section(data, "obstacles"):
        obstacles.append(
            WaveBoundary(
                shapeSpec=_shape_spec_from_dict(item),
                speed=item.get("speed"),
                randomizeSpeed=bool(item.get("randomize_speed", False)),
            )
        )

    case_config = WaveCaseConfig(
        name=str(case.get("name", "defaultCase")),
        description=str(case.get("description", "Case loaded from TOML")),
        domainBox=bool(case.get("domain_box", False)),
        domainDamping=bool(case.get("domain_damping", False)),
        smoothICs=bool(case.get("smooth_ics", False)),
        smoothIterations=int(case.get("smooth_iters", 4)),
        defaultSpeed=float(physics.get("default_speed", 1.0)),
        defaultBoundarySpeed=float(physics.get("boundary_speed", 0.01)),
        defaultObstacleSpeed=physics.get("obstacle_speeds", 0.5),
        defaultAmplitude=float(physics.get("default_amplitude", 10.0)),
        amplitudeRange=tuple(physics.get("amplitude_range", (15.0, 15.0))),
        boundarySpeedRange=tuple(physics.get("boundary_speed_range", (0.01, 0.01))),
        obstacleSpeedRange=tuple(physics.get("obstacle_speed_range", (0.3, 0.7))),
        noisyICs=bool(noise.get("enable", False)),
        noiseAmplitude=float(noise.get("amplitude", 0.1)),
        noiseType=str(noise.get("type", "perlin")),
        noiseSmoothIters=int(noise.get("smooth_iter", 4)),
        noiseSeed=int(noise.get("seed", 42)),
        sources=sources,
        obstacles=obstacles,
    )

    return simulation_config, case_config
