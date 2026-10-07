"""The pairwise artificial-viscosity "Pi" term shared by the momentum-viscosity, thermal-conductivity and
thermal-dissipation modules of `modules/dissipation`:

* `coefficients.py` -- the (alpha, beta) policy (`switchedCoefficients`, `BetaMode`),
* `pair.py`         -- `PairData`, the per-pair quantities, and `pick` (pair mean vs one-sided value),
* `terms.py`        -- one function per published formulation and the dispatch on `ViscosityTerms`,
* `dispatch.py`     -- `computePi_term` (formulation as an argument), `computePi_pair` (pair velocity as an argument,
                       formulation from the params) and `computePi_actual` (raw velocities),
* `oneSided.py`     -- `computeFrontiereQ`, CRKSPH's `Q_i` / `Q_j` from the `Frontiere2017` term.
"""

from .coefficients import switchedCoefficients
from .dispatch import computePi_actual, computePi_pair, computePi_term
from .oneSided import computeFrontiereQ

__all__ = ['computePi_actual', 'computePi_pair', 'computePi_term', 'computeFrontiereQ', 'switchedCoefficients']
