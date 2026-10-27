"""Scheme-dispatching entry point: `computeNormals` resolves a surface normal
via whichever `surfaceConfig.normalSource` is configured (color-field
gradient, lambda-gradient, Maronne, or none for `Native`), and
`detectFreeSurface` runs the configured `surfaceConfig.scheme`
(ColorField/ColorFieldGrad/Barecasco/Maronne), then dilates the resulting
mask `surfaceConfig.expansionIterations` times. Short-circuits to an
all-false mask when `surfaceConfig.active` is False.
"""

from .colorFieldDetection import detectFreeSurfaceColorField
from .colorFieldGradientDetection import detectFreeSurfaceColorFieldGradient
from .colorFieldCompute import computeColorField

from .dilation import dilateSurface
from .lambdaGrad import computeLambdaGrad, computeNormalsLambdaGrad

from .maronneNormals import computeNormalsMaronne

from .barecascoDetection import detectFreeSurfaceBarecasco
from .maronneDetection import detectFreeSurfaceMaronne
from .isolated import detectIsolated

import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, ProfilerActivity
from warpSPHCore.profiling import record_function
from typing import Optional, Union, Tuple
from warpSPHCore import *




from warpSPH.configurations.simulationConfig import SimulationConfig
from ...enumTypes import *
from ...configurations.moduleConfigurations.surfaceDetection import SurfaceDetectionConfig, SurfaceDetectionScheme, NormalSource

__all__ = ['detectFreeSurface']


def computeNormals(
    currentState: Any,
    config: SimulationConfig, schemeConfig: Any, surfaceConfig: SurfaceDetectionConfig,
    adjacency: Optional[Union[AdjacencyList, CompactHashMap]],
    renormalizationState: Optional[RenormalizationState] = None
) -> torch.Tensor:
    with record_function("[warpSPH] - (freesurface) - computeNormals"):
        if surfaceConfig.normalSource == NormalSource.LambdaGrad or surfaceConfig.normalSource == NormalSource.Maronne: 
            C, Evals, renormalizationState_ = computeRenormalizationMatrices(
                queryParticles = currentState,
                operationProperties = OperationProperties(
                    kernel = config.kernel,
                    operation = WarpOperation.Gradient,
                    operationMode = OperationDirection.AllToAll,
                    supportMode = SupportScheme.SuperSymmetric
                ),
                domain = config.domain,
                adjacency = adjacency,
                returnEigVals = True
            )
            lambdas = torch.min(torch.abs(Evals), dim=-1).values
        else:
            renormalizationState_ = renormalizationState
            lambdas = None

        if surfaceConfig.normalSource == NormalSource.ColorFieldGrad:
            colorField, colorFieldGrad = computeColorField(
                currentState,
                config, schemeConfig,
                adjacency,
            )
            normals = -torch.nn.functional.normalize(colorFieldGrad, dim = 1)
        elif surfaceConfig.normalSource == NormalSource.LambdaGrad:
            normals = computeNormalsLambdaGrad(
                currentState,
                config, schemeConfig, surfaceConfig,
                adjacency,
                renormalizationState = renormalizationState_,
                lambdas = torch.min(torch.abs(Evals), dim=-1).values
            )
        elif surfaceConfig.normalSource == NormalSource.Maronne:
            normals = computeNormalsMaronne(
                currentState,
                renormalizationState_.renormalizationMatrices,
                config, schemeConfig, surfaceConfig,
                adjacency,
            )
        elif surfaceConfig.normalSource == NormalSource.Native:
            normals = None
        else:
            raise ValueError(f"Unknown normal source: {surfaceConfig.normalSource}")
        return normals, renormalizationState_, lambdas
    

def detectFreeSurface(
    currentState: Any,
    config: SimulationConfig, schemeConfig: Any, surfaceConfig: SurfaceDetectionConfig,

    adjacency: Optional[Union[AdjacencyList, CompactHashMap]],
    renormalizationState: Optional[RenormalizationState] = None,
    returnNormals: bool = True,
    wall: Optional[Any] = None,
):
    """`wall`: the analytic boundary's `WallState`; `None`: resolved from `schemeConfig.boundaryProvider`, boundary particles when there is none."""
    with record_function("[warpSPH] - (freesurface) - detectFreeSurface"):
        if surfaceConfig.active == False:
            fsm = torch.zeros(currentState.positions.shape[0], device = currentState.positions.device, dtype = currentState.positions.dtype)
            normals = torch.zeros_like(currentState.positions)
            return (fsm, fsm, None, None) if not returnNormals else (fsm, fsm, normals, None, None)

        from ..analyticBoundary import resolveWall
        wall = resolveWall(currentState, config, schemeConfig, adjacency, wall)
        if wall is not None:
            # analytic walls: the wall continuum is added to the detector's partial sums before the decisions (modules/analyticBoundary/detector.py)
            from ..analyticBoundary.detector import detectFreeSurfaceAnalytic
            return detectFreeSurfaceAnalytic(currentState, config, schemeConfig, surfaceConfig, adjacency, returnNormals = returnNormals, wall = wall)

        normals, renormalizationState_, lambdas = computeNormals(
            currentState,
            config, schemeConfig, surfaceConfig,
            adjacency,
            renormalizationState = renormalizationState
        )
        with record_function("[warpSPH] - (freesurface) - detectFreeSurface - scheme"):
            if surfaceConfig.scheme == SurfaceDetectionScheme.ColorField:
                fsm, normals2 = detectFreeSurfaceColorField(
                    currentState,
                    config, schemeConfig, surfaceConfig,
                    adjacency,
                    returnNormals = True
                )
            elif surfaceConfig.scheme == SurfaceDetectionScheme.ColorFieldGrad:
                fsm, normals2 = detectFreeSurfaceColorFieldGradient(
                    currentState,
                    config, schemeConfig, surfaceConfig,
                    adjacency,
                    returnNormals = True
                )
            elif surfaceConfig.scheme == SurfaceDetectionScheme.Barecasco:
                fsm, normals2 = detectFreeSurfaceBarecasco(
                    currentState,
                    config, schemeConfig, surfaceConfig,
                    adjacency,
                    returnNormals = True
                )
            elif surfaceConfig.scheme == SurfaceDetectionScheme.Maronne:
                fsm, normals2 = detectFreeSurfaceMaronne(
                    currentState,
                    config, schemeConfig, surfaceConfig,
                    adjacency,
                    renormalizationState = renormalizationState_,
                    normals = normals,
                    returnNormals = True
                )
            else:
                raise ValueError(f"Unknown surface detection scheme: {surfaceConfig.scheme}")

        if surfaceConfig.normalSource == NormalSource.Native:
            normals = normals2
        if getattr(surfaceConfig, 'flagIsolated', True):
            # A row with nothing inside its support has no renormalization
            # matrix; the lambda-based schemes fall back to the identity and
            # read it as bulk (lambda = 1), although its support is all air
            # (OPEN_PROBLEMS.md §8, `isolated.py`). Exact, no threshold. It has
            # no pair interactions, so this corrects the classification (and
            # row-local consumers) rather than the neighbour sums.
            isolated = detectIsolated(currentState, config, adjacency)
            fsm = (fsm | isolated) if fsm.dtype == torch.bool else torch.where(isolated, torch.ones_like(fsm), fsm)
        with record_function("[warpSPH] - (freesurface) - detectFreeSurface - dilation"):
            fs = fsm.clone().to(dtype = currentState.positions.dtype, device = currentState.positions.device)
            for i in range(surfaceConfig.expansionIterations):
                fs = dilateSurface(
                    currentState, fs,
                    config, schemeConfig, surfaceConfig,
                    adjacency,
                    overrideIterations = 1
                )

        return (fsm, fs, renormalizationState_, lambdas,) if not returnNormals else (fsm, fs, normals, renormalizationState_, lambdas,)