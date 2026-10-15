"""Configuration for the wave-equation demo: shapes, sources, obstacles.

A wave case is described declaratively -- a list of sources and a list of
obstacles, each a :class:`ShapeSpec` -- and read from a TOML casefile by
:mod:`warpSPH.caseUtils.waveEquation.casefile`. These are re-exported from
:mod:`warpSPH.configurations` under ``Wave``-prefixed names (``WaveShapeSpec``,
``WaveCaseConfig``) because ``ShapeSpec`` and ``CaseConfig`` are too generic to
put in that namespace unqualified.

The ``randomize*`` flags and the ``*Range`` pairs exist so one casefile
describes a *family* of cases rather than a single one -- the wave system's
purpose is generating varied unstructured-data samples, so a case is a
distribution to draw from. See :mod:`warpSPH.schemes.pde.waveEquation`.
"""

import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from ..utils import *
from typing import Any, List, Optional, Union
from dataclasses import dataclass, field
from warpSPHCore import *

from dataclasses import dataclass, field

from ..geometry import SamplingScheme

@dataclass
class ShapeSpec:
    """
    Generic shape descriptor used by source and obstacle configuration.

    `kind` supports both primitive and composite shapes currently used in gencase:
    sphere, box, horizontal_line, vertical_line, vesica,
    triangle, equilateral_triangle, single_slit, double_slit, prism.
    """
    kind: str = 'sphere'
    position: Tuple[float, float] = (0.0, 0.0)
    rotation: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WaveSource:
    shapeSpec: Optional[ShapeSpec] = None

    position: Tuple[float, float] = (0.0, 0.0)
    shape: str = 'circle'  # 'point', 'line', 'area'
    radius: float = 0.1  # For circle or area sources
    size: Tuple[float, float] = (1.0, 1.0)  # For line or area sources
    magnitude: Optional[float] = None  # If None, use default from CaseConfig

    randomizeRadius: bool = False
    randomizeMagnitude: bool = False
    radiusRange: Tuple[float, float] = (0.05, 0.15)

@dataclass
class WaveBoundary:
    shapeSpec: Optional[ShapeSpec] = None

    shape: str = 'box'  # 'box', 'circle', 'line'
    position: Tuple[float, float] = (0.0, 0.0)
    size: Tuple[float, float] = (1.0, 1.0)  # For box boundaries
    radius: float = 0.1  # For circle boundaries
    speed: Optional[float] = None  # If None, use default from CaseConfig

    randomizeSpeed: bool = False

@dataclass
class CaseConfig:
    name: str = 'defaultCase'
    description: str = 'This is a default case configuration.'

    domainBox: bool = False
    domainDamping: bool = False

    smoothICs: bool = False
    smoothIterations: int = 4

    defaultSpeed: float = 1.0
    defaultBoundarySpeed: float = 0.01
    defaultObstacleSpeed: Union[float, List[float]] = 0.5
    defaultAmplitude: float = 10.0

    amplitudeRange: Tuple[float, float] = (15.0, 15.0)
    boundarySpeedRange: Tuple[float, float] = (0.01, 0.01)
    obstacleSpeedRange: Tuple[float, float] = (0.3, 0.7)

    noisyICs: bool = False
    noiseAmplitude: float = 0.1
    noiseType: str = 'perlin' 
    noiseSmoothIters: int = 4
    noiseSeed: int = 42

    sources: List[WaveSource] = ()
    obstacles: List[WaveBoundary] = ()


@dataclass
class WaveEquationConfig:
    """The wave scheme's own config: which kernel/operator modes
    `f_wave_equation` (`schemes/pde/waveEquation.py`) uses for its Laplacian.

    Fills the `SimulationConfig` slot of `WaveEquationScheme`'s `SchemeBundle`
    (`schemes/builder.py`), i.e. this is what `f_wave_equation` receives as
    `schemeConfig`. Defaults match the values `f_wave_equation` used to
    hardcode, so registering the scheme changes no behavior.
    """
    kernel: KernelFunctions = field(default=KernelFunctions.Wendland2)
    supportMode: SupportScheme = field(default=SupportScheme.SuperSymmetric)
    gradientMode: GradientScheme = field(default=GradientScheme.Difference)
    laplacianMode: LaplacianScheme = field(default=LaplacianScheme.Brookshaw)


def waveEquationConfigToDict(config: 'WaveEquationConfig') -> Dict[str, Any]:
    return {
        'kernel': config.kernel.name,
        'supportMode': config.supportMode.name,
        'gradientMode': config.gradientMode.name,
        'laplacianMode': config.laplacianMode.name,
    }


def dictToWaveEquationConfig(configDict: Dict[str, Any]) -> 'WaveEquationConfig':
    config = WaveEquationConfig()
    config.kernel = KernelFunctions[configDict['kernel']] if isinstance(configDict['kernel'], str) else configDict['kernel']
    config.supportMode = SupportScheme[configDict['supportMode']] if isinstance(configDict['supportMode'], str) else configDict['supportMode']
    config.gradientMode = GradientScheme[configDict['gradientMode']] if isinstance(configDict['gradientMode'], str) else configDict['gradientMode']
    config.laplacianMode = LaplacianScheme[configDict['laplacianMode']] if isinstance(configDict['laplacianMode'], str) else configDict['laplacianMode']
    return config