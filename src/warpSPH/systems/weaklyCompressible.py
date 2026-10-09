"""State/update/system triad for the weakly-compressible delta-SPH schemes
(`schemes/deltaSPH.py`, `schemes/divergenceFree.py`'s WCSPH path): density is
integrated via `drhodt`, with free-surface fields (`surfaceIndicators`/
`surfaceNormals`/`surfaceLambdas`) and the boundary-ghost bookkeeping
(`ghostIndices`/`ghostOffsets`) that mDBC and `rigidBody/` read and write.
`finalize` applies delta-SPH particle shifting, then integrates every rigid
body's pose and reprojects its particles (`rigidBody.integrate`/
`rigidBody.update`) each step. Density is left to the generic RK-combined
continuity update the integrator already assembled before calling this hook;
`finalize` only restores the non-fluid (mDBC) band's density, which the
integrator can't see (`DELTASPH_VALIDATION_PLAN.md` item C -- this used to be
a single-evaluation exponential override instead).
"""

from warpSPHIntegrators import *
from dataclasses import dataclass
import torch
from warpSPHCore import compileGlue
from collections.abc import Mapping
from typing import Optional
from warpSPHCore import *

from ..rigidBody.integrate import  integrateRigidBody
from ..rigidBody.update import updateBodyParticlesWCSPH

from ..modules.shifting.delta import computeDeltaShift
from ..modules.shifting.wrapper import solveShifting
from ..modules.boundaryConditions.pinned import pinnedKeepWeight, applyPinnedVelocity
from ..modules.mdbc import computeMdbcNoPenShift
from ..modules.mdbc._util import stateHasBoundaryParticles
import copy
from ..modules.gravity import computeGravity
from torch.profiler import profile, ProfilerActivity
from warpSPHCore.profiling import record_function
#: Restore the normal share of the step's gravity when the no-penetration
#: correction fires against a wall that gravity pulls *away* from (a ceiling).
#: **Off: implemented, unvalidated, left for a case that needs it.**
#:
#: `finalize` rebuilds the velocity from `vPre`, so a particle whose correction
#: fires loses that step's gravity along the corrected component. For a floor
#: that is required (otherwise it accumulates downward velocity and sinks
#: through); for a ceiling it would make a particle hover, which is the rule
#: `DELTASPH_VALIDATION_PLAN.md` 5.12 sets out: `g . n_hat < 0` keep
#: discarding, `> 0` put the gravity back.
#:
#: The rule is real, but the hovering particle it was written for turns out
#: **not** to be caused by it: on sloshingTank UID 4104 the correction is not
#: firing at all (`nopen_n = 0.000` every step, `v_norm ~ 0`, so the approach
#: gate is false and `vPre` is never used). It is pinned by a steady
#: into-the-wall acceleration of 3-12x gravity from the wall-tension
#: asymmetry instead -- see 5.13 open item 1. So this path has no known
#: reproduction, and enabling it would be a behaviour change to `finalize`
#: that nothing in the suite exercises. Turn it on together with a case where
#: a particle genuinely *approaches* a ceiling.
_RESTORE_GRAVITY_ON_CEILING_NOPEN = False


_STAT_AXIS_NAMES = ('X', 'Y', 'Z')


def _statBlock(prefix: str, x: torch.Tensor) -> dict:
    """min/max/mean/p05/p95 of a 1D tensor, `{prefix}{Min,Max,Mean,P05,P95}`.

    NaN-filled (not omitted) when `x` is empty, so a trajectory's columns stay
    the same shape whether or not any particle triggered whatever `x` counts
    this step (`np.savez`'s `row.get(k, nan)` pattern already expects this).
    """
    if x.numel() == 0:
        nan = float('nan')
        return {f'{prefix}Min': nan, f'{prefix}Max': nan, f'{prefix}Mean': nan,
                f'{prefix}P05': nan, f'{prefix}P95': nan}
    xf = x.detach().float()
    q = torch.quantile(xf, torch.tensor([0.05, 0.95], device=xf.device, dtype=xf.dtype))
    return {
        f'{prefix}Min': xf.min().item(), f'{prefix}Max': xf.max().item(),
        f'{prefix}Mean': xf.mean().item(),
        f'{prefix}P05': q[0].item(), f'{prefix}P95': q[1].item(),
    }


_STAT_SUFFIXES = ('Min', 'Max', 'Mean', 'P05', 'P95')


@compileGlue
def _accelStatBlocks(accel: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """(blocks, 5) = [min, max, mean, p05, p95] of |a| and of each axis of
    the fluid accelerations `accel`; `q` = [0.05, 0.95] on the device. One
    sort per block, quantiles bitwise `torch.quantile`'s
    (`utils/syncFree.py:quantilesFromSorted`). Pure torch (`compileGlue`)."""
    from ..utils.syncFree import quantilesFromSorted
    rows = []
    for x in [torch.linalg.norm(accel, dim=-1)] + [accel[:, a] for a in range(accel.shape[-1])]:
        xf = x.detach().float()
        pq = quantilesFromSorted(torch.sort(xf)[0], q)
        rows.append(torch.stack([xf.min(), xf.max(), xf.mean(), pq[0], pq[1]]))
    return torch.stack(rows)


class _LazyStepDiagnostics(Mapping):
    """`finalize`'s step statistics, computed from the stashed raw tensors on
    first access (see `finalize`).

    Split for the runner's graphed diagnostics: `deviceValues` is the
    sync-free part (every block over the fixed fluid subset, plus the no-pen
    active count) as 0-d device tensors; `fromHost` turns its host values into
    the mapping, computing the no-pen magnitude block -- the only part with a
    data-dependent size -- eagerly, and only on steps where the correction
    touched a particle. Values and key order are the eager ones exactly."""

    def __init__(self, accelAll, fluidMask, nopenshiftDiag):
        self.raw = (accelAll, fluidMask, nopenshiftDiag)
        self._values = None

    def deviceValues(self, fluidIndex: torch.Tensor) -> dict:
        """`fluidIndex`: the fluid rows (ascending, as the mask selects them),
        e.g. `utils.syncFree.cachedKindIndex(kinds, 0, config)`."""
        accelAll, _fluidMask, nopenshiftDiag = self.raw
        from ..utils.syncFree import deviceConstant
        dev = {}
        if fluidIndex.numel() > 0:
            accel = accelAll.index_select(0, fluidIndex)
            blocks = _accelStatBlocks(accel, deviceConstant([0.05, 0.95], torch.float32, accel.device))
            names = ['accelMag'] + [f'accel{_STAT_AXIS_NAMES[a]}' for a in range(accel.shape[-1])]
            for b, prefix in enumerate(names):
                for c, suffix in enumerate(_STAT_SUFFIXES):
                    dev[f'{prefix}{suffix}'] = blocks[b, c]
        if nopenshiftDiag is not None:
            dev['nopenshiftNActive'] = nopenshiftDiag[1].any(dim=-1).sum()
        return dev

    def fromHost(self, host: dict) -> dict:
        """The mapping's values from `deviceValues`' (host) numbers."""
        accelAll, fluidMask, nopenshiftDiag = self.raw
        stepDiag = {}
        blocks = ['accelMag'] + [f'accel{_STAT_AXIS_NAMES[a]}' for a in range(accelAll.shape[-1])]
        for prefix in blocks:
            if f'{prefix}Min' in host:
                stepDiag.update({f'{prefix}{k}': host[f'{prefix}{k}']
                                 for k in ('Min', 'Max', 'Mean', 'P05', 'P95')})
            else:
                stepDiag.update(_statBlock(prefix, accelAll[:0, 0]))
        if nopenshiftDiag is not None:
            nopenshift, active = nopenshiftDiag
            nActive = int(host['nopenshiftNActive'])
            stepDiag['nopenshiftNActive'] = nActive
            if nActive > 0:
                stepDiag.update(_statBlock(
                    'nopenshiftMag', torch.linalg.norm(nopenshift[active.any(dim=-1)], dim=-1)))
            else:
                stepDiag.update(_statBlock('nopenshiftMag', nopenshift[:0, 0]))
        return stepDiag

    def _compute(self):
        if self._values is None:
            fluidIndex = self.raw[1].nonzero().squeeze(1)
            dev = self.deviceValues(fluidIndex)
            keys = list(dev)
            vals = (torch.stack([dev[k].to(torch.float64).reshape(()) for k in keys]).cpu().tolist()
                    if keys else [])
            self._values = self.fromHost(dict(zip(keys, vals)))
        return self._values

    def __getitem__(self, key):
        return self._compute()[key]

    def __iter__(self):
        return iter(self._compute())

    def __len__(self):
        return len(self._compute())


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
    ii = getattr(adjacency, 'i', None)
    jj = getattr(adjacency, 'j', None)
    if ii is None or jj is None:
        return None
    sel = (state.kinds[ii] == 0) & (state.kinds[jj] == 1)
    if not bool(sel.any()):
        return None
    off = state.ghostOffsets[jj[sel]]
    nj = -off / off.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    acc = torch.zeros_like(state.positions)
    acc.index_add_(0, ii[sel], nj)
    return torch.nn.functional.normalize(acc, dim=-1)

__all__ = ['WeaklyCompressibleState', 'WeaklyCompressibleSystemUpdate', 'WeaklyCompressibleSystem']


@dataclass
class WeaklyCompressibleState(BaseState):
    positions : torch.Tensor = integrated('dxdt', tags=('position',))
    velocities: torch.Tensor = integrated('dvdt', tags=('velocity',))
    supports : torch.Tensor = constant(tags=('particle_support',))
    masses : torch.Tensor = constant(tags=('particle_mass',))
    # A drift field (warpSPHIntegrators/drift.py): the continuity rate's
    # velocity-linear part (`-rho div v`, reported as `drhodt_kin`) is advanced
    # with the drifts, like the positions, under the Verlet-family integrators when
    # `schemeConfig.timeCentredContinuity` is on (CEILING_STICKING_PLAN.md §3, §6).
    densities : torch.Tensor = integrated('drhodt', tags=('density',), drift_key='drhodt_kin')

    kinds : torch.Tensor = constant(tags=('particle_kind',))
    materials : torch.Tensor = constant(tags=('particle_material',))
    UIDs : torch.Tensor = constant(tags=('particle_UID',))
    UIDcounter : int = constant(tags=('particle_UIDcounter',))

    pressures : torch.Tensor = constant(tags=('damping',), default=None)
    soundspeeds : torch.Tensor = constant(tags=('soundSpeed',), default=None)
    surfaceIndicators : torch.Tensor = constant(tags=('surfaceIndicator',), default=None)
    surfaceNormals : torch.Tensor = constant(tags=('surfaceNormal',), default=None)
    surfaceLambdas : torch.Tensor = constant(tags=('surfaceLambda',), default=None)

    ghostIndices : torch.Tensor = constant(tags=('ghostIndices',), default=None)
    ghostOffsets : torch.Tensor = constant(tags=('ghostOffsets',), default=None)

    # Per-particle rigid-body acceleration at boundary/ghost rows (zero
    # elsewhere), written by `rigidBody/update.py`'s `updateBodyParticlesWCSPH`
    # every step. `None` until at least one rigid body has been updated once
    # (readers must treat `None` as "zero everywhere", the same fallback this
    # field replaces -- `modules/mdbc/english2025.py`'s `a_b` used to be
    # hardcoded to zero because no such field existed at all;
    # WCSPH_DEFAULT_CLOSEOUT_PLAN.md item D).
    boundaryAccelerations : torch.Tensor = constant(tags=('boundaryAcceleration',), default=None)

@dataclass
class WeaklyCompressibleSystemUpdate:
    dxdt: torch.Tensor = tagged(tags=('position_derivative',))
    dvdt: torch.Tensor = tagged(tags=('velocity_derivative',))
    drhodt: torch.Tensor = tagged(tags=('density_derivative',))
    passive: Optional[torch.Tensor] = tagged(tags=('passive_derivative',), default=None)
    #: the continuity part of `drhodt` (`-rho div v`, wall pairs included), the
    #: density's velocity-linear rate; None from schemes that do not report it
    drhodt_kin: Optional[torch.Tensor] = tagged(default=None)


@dataclass
class WeaklyCompressibleSystem(BaseIntegrationSystem):
    state: WeaklyCompressibleState = reference_state(tags=('physics_state',))
    adjacency: Optional[AdjacencyList] = None
    domain: Optional[DomainDescription] = None
    t: float = 0.0
    def initializeNewState(self, *args, verbose=False, **kwargs):
        state = get_reference_state(self)
        verbosePrint(verbose, f'Initializing new state [t={self.t}]')
        return WeaklyCompressibleSystem(state=state.initializeNewState(), adjacency=self.adjacency, t=self.t, domain=self.domain)

    def apply_position_update(self, update, spec: PositionUpdateSpec, **kwargs):
        return update_position(self, update, spec, 'position', 'position_derivative', 'velocity', 'velocity_derivative')
    def apply_velocity_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        return update_component(self, update, spec, 'velocity', 'velocity_derivative')
    def apply_quantity_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        # updatedsystem = update_component(self, update, spec, 'internalEnergy', 'internalEnergy_derivative')
        # updatedsystem = update_component(updatedsystem, update, spec, 'totalEnergy', 'totalEnergy_derivative')
        updatedsystem = update_component(self, update, spec, 'density', 'density_derivative')
        return updatedsystem

    def apply_state_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        # Note: DO NOT advance self.t here. Time is managed by the integrator.
        position_spec = PositionUpdateSpec(derivative_dt=spec.derivative_dt, blend=spec.blend)
        self.apply_position_update(update, position_spec, **kwargs)
        self.apply_velocity_update(update, spec, **kwargs)
        self.apply_quantity_update(update, spec, **kwargs)
        return self
    
    def drift_fields_enabled(self, schemeConfig=None, **kwargs):
        """Density as a drift field (warpSPHIntegrators/drift.py): on with
        `schemeConfig.timeCentredContinuity` (CEILING_STICKING_PLAN.md §3, §6)."""
        return bool(getattr(schemeConfig, 'timeCentredContinuity', False))

    def drift_rates(self, velocities, aux=None, config=None, schemeConfig=None, **kwargs):
        """The continuity rate `-rho div v` at this system's configuration with the
        fluid moving at `velocities` (wall rows re-derived from them by the wall
        BC, as the right-hand side does) -- the density's velocity-linear rate,
        `drhodt_kin`. Evaluated on a shallow copy; `aux` is the stage's
        `(adjacency, state)`, else this system's own adjacency."""
        from ..modules.mdbc.velocity import computeBoundaryVelocities
        from ..modules.momentum.inconsistent import computeMomentum
        adjacency = aux[0] if aux else self.adjacency
        state = copy.copy(self.state)
        fluid = (state.kinds == 0).unsqueeze(-1)
        state.velocities = torch.where(fluid, velocities, state.velocities)
        if stateHasBoundaryParticles(state, config):
            state.velocities = computeBoundaryVelocities(state, config, schemeConfig, adjacency)
        # analytic walls: `computeMomentum` adds the wall flux of the continuity equation for the velocities of this evaluation (the right-hand side adds the
        # same term for the stage velocity; the time-centred correction must see it for the mean velocity, or the density at a wall is advanced without
        # the wall's compression)
        rate = computeMomentum(state, config, schemeConfig, adjacency)
        rate = torch.where(fluid.squeeze(-1), rate, torch.zeros_like(rate))
        return WeaklyCompressibleSystemUpdate(dxdt=None, dvdt=None, drhodt=None, drhodt_kin=rate)

    def _noPenShift(self, config, schemeConfig):
        """the no-penetration correction of this state: the boundary particles' (`computeMdbcNoPenShift`) plus, with a boundary
        provider (analytic walls), the walls' (`modules/analyticBoundary/noPenetration.py`)."""
        provider = getattr(schemeConfig, 'boundaryProvider', None)
        shift = None
        if provider is None or stateHasBoundaryParticles(self.state, config):
            shift = computeMdbcNoPenShift(self.state, config, schemeConfig, self.adjacency)
        if provider is not None:
            from ..modules.analyticBoundary import analyticNoPenShift
            analytic = analyticNoPenShift(provider, self.state, config, schemeConfig, getattr(schemeConfig, '_analyticDx', None) or float(config.dx))
            shift = analytic if shift is None else shift + analytic
        return shift

    def finalize(self, initialState, dt, returnValues, updateValues, weights = ..., *args, **kwargs):
        self.adjacency = returnValues[-1][0]  # Assuming the adjacency list is the last return value from the derivative function
        # Copy the last substeps values into the current state to ensure the final state is correct
        lastState = returnValues[-1][1]  # Assuming the state is the second return value from the derivative function
        # General attributes
        self.state.supports.copy_(lastState.supports)
        # self.state.densities.copy_(lastState.densities)

        # Compressible system specific attributes (internal eneryg is integrated)
        # self.state.totalEnergies.copy_(lastState.totalEnergies)
        # self.state.entropies.copy_(lastState.entropies)
        if self.state.pressures is not None and lastState.pressures is not None:
            self.state.pressures.copy_(lastState.pressures)
        else:
            self.state.pressures = lastState.pressures.clone() if lastState.pressures is not None else None
        if self.state.soundspeeds is not None and lastState.soundspeeds is not None:
            self.state.soundspeeds.copy_(lastState.soundspeeds)
        else:
            self.state.soundspeeds = lastState.soundspeeds.clone() if lastState.soundspeeds is not None else None

        if self.state.surfaceIndicators is not None and lastState.surfaceIndicators is not None:
            self.state.surfaceIndicators.copy_(lastState.surfaceIndicators)
        else:
            self.state.surfaceIndicators = lastState.surfaceIndicators.clone() if lastState.surfaceIndicators is not None else None
            # print(f'copying surface indicators: {self.state.surfaceIndicators is not None}, {lastState.surfaceIndicators is not None}')

        if self.state.surfaceNormals is not None and lastState.surfaceNormals is not None:
            self.state.surfaceNormals.copy_(lastState.surfaceNormals)
        else:
            self.state.surfaceNormals = lastState.surfaceNormals.clone() if lastState.surfaceNormals is not None else None

        if self.state.surfaceLambdas is not None and lastState.surfaceLambdas is not None:
            self.state.surfaceLambdas.copy_(lastState.surfaceLambdas)
        else:
            self.state.surfaceLambdas = lastState.surfaceLambdas.clone() if lastState.surfaceLambdas is not None else None

        # print(self.state)

        # print(f"Finalizing state at t={self.t + dt}, dt={dt}, with {self.state.positions.shape[0]} particles.")
        config = kwargs.get('config', None)
        schemeConfig = kwargs.get('schemeConfig', None)
        if schemeConfig.shiftProperties.active:
            # shiftVector, self.adjacency = computeDeltaShift(
            #     currentState = self.state,
            #     config = config,
            #     schemeConfig = schemeConfig,
            #     domain = config.domain,
            #     adjacency = self.adjacency,
            # )
            with record_function("[warpSPH] - [deltaSPH] - solve shifting"):
                dx = solveShifting(
                    systemState = self.state,
                    config = config,
                    schemeConfig = schemeConfig,
                    adjacency = self.adjacency,
                    dt = dt,
                )
                # print(f"Applied shifting update with max shift magnitude: {dx.norm(dim=1).max().item()}")
                keep = pinnedKeepWeight(self.state, config, schemeConfig)        # a pinned band is not shifted
                if keep is not None:
                    dx = dx * keep

                du = dx / dt
                rho = self.state.densities
                u = self.state.velocities

                if schemeConfig.shiftProperties.correctdrhodt:
                    with record_function("[warpSPH] - [deltaSPH] - compute drhodt_shift"):
                        drhodt_shift = warpOperation(
                            self.state,
                            operationProperties = OperationProperties(
                                operation=WarpOperation.Divergence,
                                kernel = config.kernel, 
                                supportMode = SupportScheme.Gather,
                                operationMode = OperationDirection.AllToAll,
                                gradientMode = GradientScheme.Summation
                            ),
                            queryValues = rho.view(-1,1) * du,
                            domain = config.domain,
                            adjacency = self.adjacency
                        ) - rho * warpOperation(
                            self.state,
                            operationProperties = OperationProperties(
                                operation=WarpOperation.Divergence,
                                kernel = config.kernel, 
                                supportMode = SupportScheme.Gather,
                                operationMode = OperationDirection.AllToAll,
                                gradientMode = GradientScheme.Difference
                            ),
                            queryValues = du,
                            domain = config.domain,
                            adjacency = self.adjacency
                        )
                if schemeConfig.shiftProperties.correctdvdt:
                    with record_function("[warpSPH] - [deltaSPH] - compute dudt shift"):
                        dudt = -u * warpOperation(
                            self.state,
                            operationProperties = OperationProperties(
                                operation=WarpOperation.Divergence,
                                kernel = config.kernel, 
                                supportMode = SupportScheme.Gather,
                                operationMode = OperationDirection.AllToAll,
                                gradientMode = GradientScheme.Difference
                            ),
                            queryValues =  du,
                            domain = config.domain,
                            adjacency = self.adjacency
                        ).view(-1,1)

                        duCross = warpOperation(
                            self.state,
                            operationProperties = OperationProperties(
                                operation=WarpOperation.Divergence,
                                kernel = config.kernel, 
                                supportMode = SupportScheme.Gather,
                                operationMode = OperationDirection.AllToAll,
                                gradientMode = GradientScheme.Summation
                            ),
                            queryValues =  torch.einsum('ij,ik->ijk', u, du),
                            domain = config.domain,
                            adjacency = self.adjacency
                        )




        # `self.state.densities` already holds the RK-combined continuity update
        # by this point -- `densities` is declared `integrated('drhodt', ...)`,
        # so the generic driver's `apply_state_update` -> `apply_quantity_update`
        # -> `update_component` pass (`finalizeSystem` runs it before calling
        # this hook) has already summed `rho^n + sum_i(b_i * dt * drhodt_i)` over
        # every RK stage, the same combination positions/velocities get. Leave
        # it alone for fluid particles.
        #
        # Previously this was overwritten with a single-evaluation exponential
        # map `rho^n * exp(dt * drhodt_lastStage / rho_lastStage)` -- exact only
        # for a frozen rate (a symplectic/semi-implicit-Euler shape), not for a
        # multi-stage RK where combining the stage rates *is* the method: it
        # used just the *last* stage's rate from the *initial* density, so
        # density was ~1st order while position/velocity were the integrator's
        # full order, and the convex `exp` rectifies an oscillatory drhodt into
        # a net density gain (DELTASPH_VALIDATION_PLAN.md item C / sloshingTank
        # acoustic ringing). The commented-out Padé(1,1) form below it was
        # equally a single-evaluation override and was already dead (`epsilon`
        # computed, never read).
        #
        # Non-fluid (boundary/rigid) particles: `enforceUpdates` masks their
        # `drhodt` to zero, so the generic pass leaves them at rho^n, not at the
        # mDBC value `deltaSPH_step` computed mid-step. Restore that explicitly.
        midRho = returnValues[-1][1].densities
        self.state.densities = torch.where(self.state.kinds != 0, midRho, self.state.densities)

        # Same guarantee for position. `integrated('dxdt', ...)`
        # (`warpSPHIntegrators/fields.py`) declares `fluid_only=True` on this
        # field, but that flag is never actually consulted by any generic
        # integration path -- it's dead metadata. `schemes/deltaSPH.py` masks
        # `update.dxdt` to zero for `kinds != 0` (belt) after every stage, which
        # covers the explicit/RK integrators, but `symplecticEuler`'s second
        # position half-step (`semi_implicit_position_step`,
        # `warpSPHIntegrators/verlet.py`) reads the *raw* current velocity
        # directly (`x += dt/2 * v_current`), not the masked `dxdt` -- so a
        # boundary particle carrying a nonzero BC-prescribed velocity (a moving
        # wall's Dirichlet condition, e.g. `cases/lidDrivenCavity.py`) drifts by
        # `velocity * dt/2` **every step**, unboundedly, even though that
        # velocity exists only to drive the SPH force sums, never to move the
        # wall (suspender). Measured: 370 lid-band ghost/boundary particles
        # drifting linearly, 1.5 domain-widths by t=3 at `lidVelocity=1`,
        # wrecking the lattice the pressure/density estimate at the lid
        # interface depends on -- the actual cause of the corner-seeded,
        # lid-line shear instability under `--integrationScheme
        # symplecticEuler` this was traced from (not a genuine corner
        # singularity, though `regularizeLid` still helps by shrinking the
        # velocity, hence the drift rate, near the corners).
        self.state.positions = torch.where(
            (self.state.kinds != 0).unsqueeze(-1),
            initialState.state.positions, self.state.positions)
        # Defensive floor only -- catches an actual sign flip / NaN from a
        # pathological step, not part of the routine update (normal weakly-
        # compressible density stays within a few percent of rho0).
        self.state.densities = torch.nan_to_num(
            self.state.densities, nan=schemeConfig.fluid.restDensity,
            posinf=schemeConfig.fluid.restDensity, neginf=schemeConfig.fluid.restDensity
        ).clamp_min(0.05 * schemeConfig.fluid.restDensity)

        # Lone-particle density reset (`schemeConfig.loneDensityReset`,
        # MDBC_CONTACT_LINE_PLAN.md §12): detected on the last stage's state
        # with the adjacency it was evaluated with (positions and pair list
        # consistent), applied to the final density below.
        loneMask = None
        if getattr(schemeConfig, 'loneDensityReset', False):
            from ..modules.surfaceDetection import detectNoFluidNeighbours
            loneMask = (lastState.kinds == 0) & detectNoFluidNeighbours(lastState, config, self.adjacency)

        if schemeConfig.shiftProperties.active:
            if schemeConfig.shiftProperties.correctdrhodt:
                self.state.densities += drhodt_shift * dt
            if schemeConfig.shiftProperties.correctdvdt:
                self.state.velocities += (dudt + duCross) * dt
            self.state.positions += dx

        if loneMask is not None:
            self.state.densities = torch.where(
                loneMask, torch.full_like(self.state.densities, schemeConfig.fluid.restDensity),
                self.state.densities)

        # mDBC no-penetration correction, DualSPHysics placement: once per real
        # step, here with the other post-integration corrections, rather than as
        # a force re-evaluated at every RK sub-stage (`DELTASPH_VALIDATION_PLAN`
        # 5.9). `JSphGpuSimple_ker.cu`'s `MDBC2_NoPen` blocks do exactly this --
        # `v_new = v^n + nopenshift` **replacing** the integrated component, and
        # the displacement recomputed from it -- so this is a velocity
        # replacement, not another acceleration summed with pressure and
        # gravity. Applied per component, only where the correction is non-zero,
        # and only to fluid particles (the wall band's own motion comes from the
        # BC machinery).
        #
        # `'impulse'` (CEILING_STICKING_PLAN.md §7): the same correction as a
        # contact impulse on the *integrated* velocity, `v^{n+1} = v + shift(v)`,
        # with x and rho left as integrated (a configuration is continuous
        # across an impulse). The replacement above rebuilds from `v^n`, so it
        # discards the step's pressure/gravity impulse along the corrected
        # component while rho keeps the compression the approach booked: every
        # step it fires on a particle driving into a wall, the pressure work
        # that should have braked it is created as energy. Measured on a lone
        # ceiling rider (Marrone 3.1 seed 2, uid 80): that is the whole growth of
        # its ringing, +0.2 ... +11 J/kg per cycle, released at 10 m/s. As an
        # impulse the normal approach velocity becomes (1 - factor) v_n, a
        # restitution e = 1 - factor, so |v_n| cannot grow for factor in [0, 2]
        # (every row on the fluid side of a wall row).
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
                            if _RESTORE_GRAVITY_ON_CEILING_NOPEN else None)
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

        for rigidBody in schemeConfig.rigidBodies:
            dudt, dwdt = 0, 0
            if getattr(rigidBody, 'dynamic', False):
                # a free analytic body: the fluid's load (booked by the scheme's right-hand side) and gravity drive it (acceleration of the centre of mass and of the rotation)
                if rigidBody.load is None:
                    raise RuntimeError('a dynamic rigid body needs the loads of the scheme\'s right-hand side (analytic boundaries only)')
                g = computeGravity(self.state, config, schemeConfig, self.adjacency)
                g = (g.reshape(-1, g.shape[-1])[0] if g.dim() > 1 else g).to(rigidBody.centerOfMass.dtype)
                load = rigidBody.load.sum(0).to(rigidBody.centerOfMass.dtype)
                dudt = load[:2] / rigidBody.mass + g
                dwdt = load[2] / rigidBody.inertia
            rigidBody = integrateRigidBody(rigidBody, dudt, dwdt, dt)
            if getattr(rigidBody, 'representation', None) is None:                 # an analytic body has no particles to move (and the update reads host values, which a graph capture forbids)
                self.state = updateBodyParticlesWCSPH(self.state, rigidBody)

        # Per-step diagnostics: the *net* fluid acceleration actually applied
        # this step (magnitude + per axis), and -- only under
        # `mdbcNoPenShiftMode` 'finalize' or 'impulse', where `nopenshiftDiag` above was
        # set regardless of whether any particle ended up `active` -- how many
        # fluid particles the no-pen correction touched and by how much.
        # Stashed on `self` (the same object every case's `diagnostics(ctx,
        # state)` receives as `state`) rather than returned, so it rides every
        # trajectory row for free (`cases/weaklyCompressible.py
        # stepAccelerationDiagnostics`) without every case needing to know
        # this system's internals. Added after the overnight batch
        # (`DELTASPH_VALIDATION_PLAN.md`) made "was it nopenshift?" a
        # recurring question across the lid-driven cavity, Marrone 3.1 and
        # Marrone 3.4 investigations that otherwise needed a one-off probe
        # script and a from-scratch re-run each time to answer.
        # The per-step statistics (quantiles over the fluid rows and over the
        # no-pen-active subset) need host syncs and data-dependent sizes, and
        # nothing in the step reads them -- only the case diagnostics do. So
        # the raw full-size tensors are kept and the same statistics are
        # computed on first read (`_LazyStepDiagnostics`), outside any
        # captured step graph. Values are unchanged (elementwise ops commute
        # with the row gather).
        fluidMask = self.state.kinds == 0
        accelAll = (self.state.velocities - initialState.state.velocities) / dt
        self.stepDiagnostics = _LazyStepDiagnostics(accelAll, fluidMask, nopenshiftDiag)

        # Information for artificial viscosity switches
        # self.state.divergence.copy_(lastState.divergence)
        # self.state.alpha0s.copy_(lastState.alpha0s)
        # self.state.alphas.copy_(lastState.alphas)

        # print(self.state)

        # print(f'Surface particles: {self.state.surfaceIndicators.sum().item()} / {self.state.surfaceIndicators.shape[0]} ({100 * self.state.surfaceIndicators.sum().item() / self.state.surfaceIndicators.shape[0]:.2f}%)')

        applyPinnedVelocity(self.state, kwargs.get('config', None), kwargs.get('schemeConfig', None))
        return super().finalize(initialState, dt, returnValues, updateValues, weights, *args, **kwargs)
    