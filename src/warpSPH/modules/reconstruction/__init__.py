"""Pair-velocity policies for the pairwise dissipation operator (AV_PLAN Phases 1 / 3): which velocity
difference `u_ij` the artificial-viscosity term sees.

* `pairVelocity.py` -- `rawPairVelocity` (the historical behaviour), `linearPairVelocity` (the midpoint
  extrapolation, Garcia-Senz & Cabezon 2026 Eqs. 11-12), `limitedPairPhi` and the policy dispatch
  `reconstructPairVelocity`;
* `limiters.py`     -- the van Leer-like limiter and the close-pair taper, moved here from `modules/crk`;
* `gradient.py`     -- `computeVelocityJacobian`, the (optionally M^-1-corrected) velocity Jacobian, and the
  Balsara factor from it (`balsaraFromJacobian`);
* `diagnostics.py`  -- `computePairPhiMean`, the per-particle mean reconstruction factor (the detector map).

CRKSPH (`modules/crk`) reconstructs through these with its own CRK-corrected gradient; the Monaghan host
through `DiffusionParameters.velocityPairPolicy`."""

from .pairVelocity import rawPairVelocity, linearPairVelocity, limitedPairPhi, reconstructPairVelocity
from .limiters import limiterVL, limiterPsi, computeVanLeer, crkLimiter
from .pairState import reconstructPairScalar, reconstructPairRiemannStates, pairRiemannStates, limitedIncrement, reconstructPairIncrements, reconstructPair1D
from .gradient import (stateGradientInputs, computeStateGradients, needsStateGradients, stateGradientArguments, computeVelocityJacobian, velocityTensorArguments, needsJacobian, needsBalsara,
                       balsaraFromJacobian, reconstructionInputs, requireRawPairVelocity)
from .diagnostics import computePairPhiMean

__all__ = ['rawPairVelocity', 'linearPairVelocity', 'limitedPairPhi', 'reconstructPairVelocity',
           'limiterVL', 'limiterPsi', 'computeVanLeer', 'reconstructPairScalar', 'reconstructPairRiemannStates', 'pairRiemannStates', 'limitedIncrement', 'reconstructPairIncrements', 'reconstructPair1D', 'crkLimiter', 'computeVelocityJacobian', 'velocityTensorArguments', 'needsJacobian', 'needsBalsara', 'balsaraFromJacobian',
           'reconstructionInputs', 'requireRawPairVelocity', 'computePairPhiMean',
           'computeStateGradients', 'needsStateGradients', 'stateGradientArguments', 'stateGradientInputs']
