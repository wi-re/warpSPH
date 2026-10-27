"""compSPH: Monaghan-style compressible SPH pressure/viscosity forces and thermal energy.

Pressure+artificial-viscosity acceleration (`accel`), the matching internal-energy
rate (`dudt`), the pairwise energy-partition factor used to keep momentum and
energy updates consistent (`balance`), and the compatible internal-energy update that folds each
Runge-Kutta stage's pair work into one running sum (`pairWork`). `operators` is the
(state, config, schemeConfig, adjacency, gradH) interface the schemes call; the `*Warp`
functions are the low-level launchers underneath.
"""

from .accel import computeCompSPHAccelWarp
from .dudt import computeCompSPHdudtWarp
from .balance import computeCompSPHBalanceTermWarp
from .operators import computeCompSPHAccel, computeCompSPHdudt, computeCompSPHBalanceTerm
from .pairWork import PairWork, beginPairWork, accumulateStage, energyIncrement

__all__ = ['computeCompSPHAccel', 'computeCompSPHdudt', 'computeCompSPHBalanceTerm', 'computeCompSPHAccelWarp', 'computeCompSPHdudtWarp', 'computeCompSPHBalanceTermWarp', 'PairWork', 'beginPairWork', 'accumulateStage', 'energyIncrement']