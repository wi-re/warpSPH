"""Completeness/bulk-gated pressure-gradient correction; existing policy unchanged."""
import torch
from warpSPHCore import OperationProperties, WarpOperation, SupportScheme, RenormalizationState, warpOperation

def _gatedPressureRenormalization(currentState, config, adjacency, renormalizationState):
    """The renormalization matrices of the pressure force, gated per particle (DELTASPH_VALIDATION_PLAN sections 5.23 / 5.32).

    Applying ``Li`` unconditionally makes Marrone 3.1's free-surface peak 8.8x WORSE: ``Li`` comes from ``detectFreeSurface``'s covariance fit, which is most
    ill-conditioned exactly in the sparse / disordered neighbourhoods where the pressure-force truncation artifact lives. DualSPHysics' advanced-shifting
    extension instead gates the renormalized form on kernel-sum COMPLETENESS (``poup1 > 0.95``, a plain partition-of-unity sum) AND bulk classification, falling
    back to the raw gradient everywhere else (every free-surface particle included). Reproduced here on the matrix itself (per particle): the identity matrix for
    a gated-off particle makes ``matmul(Li, gradw_ij) == gradw_ij``, with no kernel change. Same support scheme as ``computePressureForceSurfaceAware``,
    ``surfaceIndicators`` as the bulk / free-surface classification."""
    completenessProps = OperationProperties(
        kernel=config.kernel, operation=WarpOperation.Interpolate,
        supportMode=SupportScheme.SuperSymmetric)
    kernelCompleteness = warpOperation(
        currentState, completenessProps, domain=config.domain,
        referenceValues=torch.ones_like(currentState.densities),
        adjacency=adjacency)
    wellConditioned = (kernelCompleteness > 0.95) & (currentState.surfaceIndicators == 0)
    Li = renormalizationState.renormalizationMatrices
    dim = currentState.positions.shape[1]
    identity = torch.eye(dim, dtype=Li.dtype, device=Li.device)
    gatedMatrices = torch.where(wellConditioned.view(-1, 1, 1), Li, identity)
    return RenormalizationState(renormalizationMatrices=gatedMatrices)
