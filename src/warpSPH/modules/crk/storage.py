"""Storage routing only; numerical policy and fallback implementations are unchanged."""
import torch
from ...enumTypes import EnergyScheme


def useRecomputedWork(system, state, scheme, wall, stage, index):
    """Default native eager path; retain the reference route in other contexts."""
    return (getattr(scheme, 'crkPairStorage', 'recompute') == 'recompute'
            and scheme.compatibleEnergy and scheme.energyScheme == EnergyScheme.CRK
            and wall is None
            and getattr(scheme, 'crkEnergyPartition', 'native') == 'native'
            and getattr(scheme, 'crkEnergyPredictor', 'raw') == 'raw'
            and getattr(scheme, 'crkEvolution', 'summation') == 'summation'
            and getattr(scheme, 'crkDensityPolicy', 'legacy') == 'legacy'
            and getattr(scheme, 'crkViscosityGradient', 'current') == 'current'
            and getattr(scheme, 'materials', None) is None
            and getattr(scheme, 'stepMode', 'eager') in (None, 'eager')
            and system.pairWork is not None and system.pairWork.weights is not None
            and index is not None
            and not any(v.requires_grad or torch.autograd.forward_ad.unpack_dual(v).tangent is not None
                        for v in vars(state).values() if isinstance(v, torch.Tensor))
            and not stage.volumes.requires_grad)


def derivativeInputs(*values):
    """An external force or differentiable dt may introduce AD after force routing."""
    return any(isinstance(v, torch.Tensor) and
               (v.requires_grad or torch.autograd.forward_ad.unpack_dual(v).tangent is not None)
               for v in values)
