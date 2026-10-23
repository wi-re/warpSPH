"""Reusable matrix-free IISPH pressure projection.

All fields and neighbor reductions stay on the state's device. The operator
is A = C F: F maps q=p/rho0 to acceleration and C maps velocity to the rate
of summation density/rho0 using *rest* volumes m/rho0. Its diagonal includes
the contact extrapolation, when present. No sparse pressure matrix is built.
"""

from dataclasses import dataclass
from copy import copy
from types import SimpleNamespace
import math

import torch
import warp as wp
from warpSPHCore.linalg import CheckSchedule, runStationary
from warpSPHCore.type_config import scalar_t, get_torch_precision
from .wp_iisph import kernels


class IISPHPairs:
    """Kernel geometry over the Core's row-ordered live neighbor list."""

    def __init__(self, state, config, adjacency, reference=None):
        reference = state if reference is None else reference
        self.n = len(state.positions)
        self.device = state.positions.device
        self.kernels = kernels(state.positions.shape[1])
        self.j = adjacency.livePairs()[1].to(torch.int64).contiguous()
        self.starts = adjacency.edgeOffsets.to(torch.int64).contiguous()
        self.counts = adjacency.numNeighbors.to(torch.int64).contiguous()
        self.gradient = torch.empty((len(self.j), state.positions.shape[1]),
                                    dtype=state.positions.dtype, device=self.device)
        self.weights = torch.empty_like(self.gradient[:, 0]).contiguous()
        self.norm = 1.0
        if config.calibrateNormalization:
            from warpSPHCore.util.latticeDensity import latticeDensityFactor
            self.norm = latticeDensityFactor(config.kernel, float(config.n_h), state.positions.shape[1])
        self.geometryParameters = [self.kernels['V'](*(config.domain.max-config.domain.min).tolist()),
                                   self.kernels['I'](*[int(v) for v in config.domain.periodic.tolist()]),
                                   config.kernel.value, scalar_t(self.norm)]
        self.launch('geometry', [state.positions, reference.positions, state.supports,
                    adjacency.livePairs()[0].to(torch.int64).contiguous(), self.j,
                    *self.geometryParameters, self.gradient, self.weights], len(self.j))

    def launch(self, name, args, n=None):
        values = []
        for value in args:
            if isinstance(value, torch.Tensor):
                value = value.contiguous()
                dtype = self.kernels['V'] if value.ndim == 2 else None
                values.append(wp.from_torch(value, dtype=dtype))
            else:
                values.append(value)
        wp.launch(self.kernels[name], dim=self.n if n is None else n,
                  inputs=values, device=str(self.device))

    def base(self):
        return [self.j, self.starts, self.counts, self.gradient]


@dataclass
class IISPHResult:
    pressure: torch.Tensor
    acceleration: torch.Tensor
    iterations: int
    error: float
    termination: str
    source: torch.Tensor


def iisphViscosity(state, config, schemeConfig, adjacency):
    """Configured two-sided fluid viscosity with free-slip static walls.

    The shared viscosity divides its kinematic coefficient by density. A
    normalized shallow view retains physical-unit invariance and leaves the
    physical masses/density used by gravity, forcing and pressure untouched.
    """
    from warpSPHCore import OperationDirection
    from ..deltaSPH import computeVelocityDiffusion
    view = copy(state)
    view.masses = state.masses / schemeConfig.fluid.restDensity
    view.densities = state.densities / schemeConfig.fluid.restDensity
    return computeVelocityDiffusion(view, config, schemeConfig, adjacency,
        approachOnly=False, wall=False, operationMode=OperationDirection.FluidToFluid)


class IISPHOperator:
    """Fixed-position, fixed-common-support single-fluid pressure operator.

    kind 0 are fluid unknowns; kind 1 are stationary sampled solids at rest
    density, with supplied physical masses. Analytic solids supply B and G
    instead. Moving bodies, ghost samples and differentiable solves are not
    supported. These restrictions are checked rather than silently changing
    the pressure law. 2D and 3D particle/periodic operators use the same code.
    """

    def __init__(self, state, config, adjacency, rho0, *, boundary=None):
        if not math.isfinite(rho0) or rho0 <= 0 or len(state.positions) == 0:
            raise ValueError("IISPH requires a nonempty state and positive rest density")
        if state.positions.shape[1] not in (2, 3):
            raise NotImplementedError("IISPH supports 2D and 3D")
        fields = [state.positions, state.supports, state.masses, state.velocities]
        if any(v.requires_grad for v in fields):
            raise NotImplementedError("IISPH has no autograd contract")
        if any(v.dtype != get_torch_precision() or v.device != state.positions.device for v in fields):
            raise ValueError("IISPH fields must share the bootstrapped precision and device")
        if not bool((torch.isfinite(state.masses) & (state.masses > 0)).all()):
            raise ValueError("IISPH requires finite positive particle masses")
        if not bool((state.supports == state.supports[0]).all()) or not bool((state.supports > 0).all()):
            raise NotImplementedError("IISPH requires fixed common support")
        if not bool(((state.kinds == 0) | (state.kinds == 1)).all()):
            raise NotImplementedError("IISPH supports fluid and static boundary samples; no ghost rows")
        self.fluid = state.kinds == 0
        if not bool(self.fluid.any()):
            raise ValueError("IISPH requires fluid particles")
        if bool((state.velocities[~self.fluid] != 0).any()):
            raise NotImplementedError("IISPH supports static bodies only")
        self.state, self.config, self.rho0 = state, config, rho0
        self.pairs = IISPHPairs(state, config, adjacency)
        self.volume = (state.masses / rho0).contiguous()
        self.kinds = state.kinds.to(torch.int32).contiguous()
        self.densityRatio = torch.empty_like(state.masses)
        self.G = torch.empty_like(state.positions)
        self.c = torch.empty_like(state.positions)
        self.diagonal = torch.empty_like(state.masses)
        B = torch.zeros_like(state.masses) if boundary is None else boundary.density
        G = torch.zeros_like(state.positions) if boundary is None else boundary.gradient
        sampled = boundary is None or boundary.sampled
        self.pairs.launch('coefficients', self.pairs.base() + [self.pairs.weights,
            self.volume, self.kinds, sampled, B, G, self.densityRatio, self.G, self.c, self.diagonal])
        self.constantAcceleration = torch.zeros_like(state.positions)
        self.contact = None
        if boundary is not None:
            boundary.attach(self)

    def densityRate(self, field):
        """C(field), for a static wall; positive means compression."""
        out = torch.empty_like(self.volume)
        self.pairs.launch('rate', self.pairs.base() + [self.volume, self.kinds, self.G, field, out])
        return out

    def acceleration(self, q):
        """Linear F(q), excluding the prescribed gravity extension."""
        contact = torch.zeros_like(q) if self.contact is None else self.contact.value(q)
        out = torch.empty_like(self.G)
        self.pairs.launch('force', self.pairs.base() + [self.volume, self.kinds,
            self.densityRatio, self.c, self.G, contact, q, out])
        return out

    def apply(self, q):
        return self.densityRate(self.acceleration(q))

    def solve(self, velocity, previousPressure, dt, policy):
        """IISPH's positive Jacobi solve and signed active-pressure statistic.

        The statistic uses the new diagonal and old off-diagonal iterate,
        averaged over *all fluid rows*. Tolerance is percent of rho0. The
        previous physical pressure is halved; it is not dt-scaled DFSPH kappa.
        """
        policy.validate()
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("IISPH requires a finite positive timestep")
        if velocity.requires_grad or previousPressure.requires_grad:
            raise NotImplementedError("IISPH has no autograd contract")
        dt2 = dt * dt
        b = (1 - self.densityRatio - dt * self.densityRate(velocity)
             - dt2 * self.densityRate(self.constantAcceleration))
        b = torch.where(self.fluid, b, 0)
        q = torch.where(self.fluid, 0.5 * previousPressure / self.rho0, 0)
        diagonal = dt2 * self.diagonal
        valid = self.fluid & (diagonal.abs() > 1.e-9)
        denominator = torch.where(valid, diagonal, 1)
        omega = policy.relaxationFactor

        def iteration(old, _):
            off = dt2 * self.apply(old) - diagonal * old
            new = torch.where(valid, ((1-omega)*old + omega*(b-off)/denominator).clamp_min(0), 0)
            errors = torch.where(new != 0, self.rho0*(diagonal*new + off-b), 0)
            return new, errors[self.fluid].mean(), None

        schedule = CheckSchedule(SimpleNamespace(), 'iisph', 'every',
                                 policy.minIterations-1, policy.maxIterations)
        result = runStationary(iteration, q, maxIters=policy.maxIterations,
            tolerance=0.01*policy.maxErrorPercent*self.rho0, inclusive=True, schedule=schedule)
        return IISPHResult(result.x*self.rho0,
            self.acceleration(result.x)+self.constantAcceleration,
            result.iterations, result.stats[-1], result.termination, b)
