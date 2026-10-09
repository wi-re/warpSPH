"""CRKSPH: an approaching pair that closes to r ~ 1e-3 dx must not blow the run up (OPEN_PROBLEMS.md §15).

Frontiere et al. 2017 Eq. (69) regularises the viscosity switch as
`mu = min(0, v_hat . eta / (eta . eta + eps^2))` with `eta = x / h` dimensionless and `eps^2 = 1e-2`. The kernel
used to add `1e-7 h^2` (~1e-11) to the dimensionless `eta . eta`, so `mu ~ v / |eta|` diverged as a pair closed and one
right-hand-side evaluation gave that pair an acceleration ~ 1e4-1e5 (Sod 2D on the same lattice, step 164).
"""
import torch

from warpSPH.cases.sodND import sod2dCase
from warpSPH.runner import run


def _sod2dSameLattice(nSteps):
    return run(sod2dCase, scheme='CRKSPH', nSteps=nSteps, plot=False, store=False, quiet=True, progress=False,
               supportMode='KernelMeanSymmetric', params=dict(equalMass=False))


def test_massJumpSod2dSurvivesPastTheFormerBlowUp(runtime):
    """Same lattice on both sides (dense side 4x heavier per particle): the first light columns pile up at the
    contact and pairs of them coincide by step ~ 90-160; the run used to die at step 164 (|v| ~ 1e3, u ~ -5e5)."""
    result = _sod2dSameLattice(220)
    st = result.state.state
    assert torch.isfinite(st.velocities).all() and torch.isfinite(st.internalEnergies).all()
    # the exact solution's largest speed is the contact/plateau velocity 0.61 (units of the case); allow ringing
    assert float(st.velocities.norm(dim=-1).max()) < 2.0
    assert float(st.internalEnergies.min()) > 0.0


def test_massJumpSod2dConservesEnergy(runtime):
    a = _sod2dSameLattice(2)
    b = _sod2dSameLattice(220)

    def total(r):
        s = r.state.state
        return float((s.masses * (s.internalEnergies + 0.5 * (s.velocities ** 2).sum(-1))).sum())
    assert abs(total(b) - total(a)) < 1e-4 * abs(total(a))
