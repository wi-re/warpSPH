"""Existing post-integration contact policies, separate from RK staging."""
import torch
from warpSPHCore.profiling import record_function
from . import computeMdbcNoPenShift
from ._util import stateHasBoundaryParticles
from ..gravity import computeGravity

def _meanBoundaryNormal(state, adjacency):
    """Mean inward normal of each fluid particle's boundary neighbours, from
    their ghost offsets (`n_j = -offset_j / |offset_j|`, the same normal the
    no-penetration kernel reflects along).

    Returns `None` when the neighbour pairs are not available (a hash-map
    adjacency), in which case the caller leaves its behaviour unchanged rather
    than guessing an orientation. The shift vector itself is *not* a usable
    proxy: its `factor = -4 ratio + 3` turns negative at deep contact, so the
    shift can point back into the wall (`scripts/probe_nopenShiftResponse.py`
    measures the reversal past ~1.25 dx).
    """
    if not hasattr(adjacency, 'livePairs'):
        return None
    ii, jj = adjacency.livePairs()
    sel = (state.kinds[ii] == 0) & (state.kinds[jj] == 1)
    if not bool(sel.any()):
        return None
    off = state.ghostOffsets[jj[sel]]
    nj = -off / off.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    acc = torch.zeros_like(state.positions)
    acc.index_add_(0, ii[sel], nj)
    return torch.nn.functional.normalize(acc, dim=-1)

def noPenShift(self, config, schemeConfig):
    """the no-penetration correction of this state: the boundary particles' (`computeMdbcNoPenShift`) plus, with a boundary
    provider (analytic walls), the walls' (`modules/analyticBoundary/noPenetration.py`)."""
    provider = getattr(schemeConfig, 'boundaryProvider', None)
    shift = None
    if provider is None or stateHasBoundaryParticles(self.state, config):
        shift = computeMdbcNoPenShift(self.state, config, schemeConfig, self.adjacency)
    if provider is not None:
        from ..analyticBoundary import analyticNoPenShift
        analytic = analyticNoPenShift(provider, self.state, config, schemeConfig, getattr(schemeConfig, '_analyticDx', None) or float(config.dx))
        shift = analytic if shift is None else shift + analytic
    return shift

def applyNoPenetration(self, initialState, dt, config, schemeConfig, *, restoreGravity=False):
    nopenshiftDiag = None    # (nopenshift, active) for the diagnostics block below
    _noPenMode = getattr(schemeConfig, 'mdbcNoPenShiftMode', 'derivative')
    if _noPenMode == 'impulse':
        with record_function("[warpSPH] - [deltaSPH] - no-pen shift (impulse)"):
            nopenshift = self._noPenShift(config, schemeConfig)
            active = (nopenshift != 0) & (self.state.kinds == 0).unsqueeze(-1)
            nopenshiftDiag = (nopenshift, active)
            self.state.velocities = torch.where(
                active, self.state.velocities + nopenshift, self.state.velocities)
    if _noPenMode == 'finalize':
        with record_function("[warpSPH] - [deltaSPH] - no-pen shift (finalize)"):
            nopenshift = self._noPenShift(config, schemeConfig)
            active = (nopenshift != 0) & (self.state.kinds == 0).unsqueeze(-1)
            nopenshiftDiag = (nopenshift, active)
            # Unconditional (was `if bool(active.any()):`): every write below
            # is a torch.where on `active`, so with no active row the state
            # comes out unchanged -- same values, no host sync.
            if True:
                vPre = initialState.state.velocities
                xPre = initialState.state.positions
                # `vPre + nopenshift` rebuilds from the *start-of-step*
                # velocity, so every step the correction fires the particle
                # loses that step's gravity along the corrected component.
                # For a floor that is required -- otherwise it accumulates
                # downward velocity and sinks through -- but for a ceiling
                # it is what makes a particle hover: measured on
                # sloshingTank, UID 4104 sat at y = 0.50700 with vy ~ 0 for
                # the final 10,000 steps, healthy rho and zero fluid
                # neighbours (`DELTASPH_VALIDATION_PLAN.md` 5.12 / 5.13).
                #
                # The orientation decides it. With `n_hat` the mean inward
                # normal of the boundary neighbours: `g . n_hat < 0` means
                # gravity presses into the wall (fluid resting on a floor)
                # and the discard is correct; `> 0` means gravity pulls
                # away from the wall (a ceiling) and the step's gravity has
                # to be put back, or nothing ever makes the particle fall.
                # Restoring only the normal share, and only in that case,
                # leaves the floor path untouched.
                gravityRestore = torch.zeros_like(nopenshift)
                nHat = (_meanBoundaryNormal(self.state, self.adjacency)
                        if restoreGravity else None)
                if nHat is not None:
                    g = computeGravity(self.state, config, schemeConfig,
                                       self.adjacency)
                    gProj = (g * nHat).sum(dim=-1, keepdim=True)
                    gravityRestore = torch.where(
                        gProj > 0, dt * gProj * nHat,
                        torch.zeros_like(gravityRestore))
                vNew = torch.where(active, vPre + nopenshift + gravityRestore,
                                   self.state.velocities)
                self.state.positions = torch.where(
                    active, xPre + vNew * dt, self.state.positions)
                self.state.velocities = vNew

    return nopenshiftDiag
