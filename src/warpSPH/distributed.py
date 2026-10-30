"""Eager replicated-state process boundaries for the bounded row-chunk backend.

Select this path by explicitly constructing a chunked adjacency with a
RowChunkProcessGroup. It does not initialize processes or change runner defaults.
"""
from hashlib import sha256
import math

import torch

from warpSPHCore.radiusSearch import ChunkedAdjacency

CHUNKED_SCHEMES=('Monaghan','CRKSPH','CompSPH','GSPH','InutsukaGSPH')


def processGroup(system):
    adjacency = getattr(system, 'adjacency', None)
    return adjacency.communication if isinstance(adjacency, ChunkedAdjacency) else None


def _stateTensors(system):
    return {k:v for k,v in vars(system.state).items() if isinstance(v, torch.Tensor)}


def validateSourceClock(system, communication):
    def metadata():
        source = system.sourceTerms
        from .modules.meshless.sources import ConservedSources
        from .modules.forcing import FourierOU
        if not isinstance(source, ConservedSources) or not isinstance(source.acceleration, FourierOU):
            raise NotImplementedError('Distributed body sources currently require ConservedSources with FourierOU.')
        field = source.acceleration
        # Materialize the same prescribed physical clock before hashing knots.
        field.accelerationBound(float(system.t))
        state = field.state_dict()
        def digest(t):
            return sha256(t.contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
        return (float(system.t), state['parameters'], digest(state['knots']), digest(state['generator']),
                source.removeMeanAcceleration, source.targetInternalEnergy, source.coolingTime)
    contract = communication.localPhase('source:clock-local', metadata)
    communication.agree('source:clock', contract)


def distributedRunStart(ctx, system):
    communication = processGroup(system)
    if communication is None:
        return None
    def validate():
        communication.validateAdjacency(system.adjacency)
        if getattr(ctx.scheme, 'name', ctx.scheme) not in CHUNKED_SCHEMES:
            raise NotImplementedError('Distributed row chunks support eager Monaghan/CRKSPH/CompSPH/GSPH/InutsukaGSPH only.')
        if (ctx.scratch.get('execution') != 'eager' or ctx.spec.store or ctx.spec.plot
            or ctx.spec.video or (ctx.spec.resumeFrom and not ctx.scratch.get('_rowChunkRestartVerified'))):
            raise NotImplementedError('Distributed proof of concept requires eager execution, automatic output disabled and validated chunked resume.')
        return (getattr(ctx.scheme, 'name', ctx.scheme), ctx.spec.nSteps, ctx.spec.tLimit,
                str(ctx.spec.integrationScheme), str(ctx.schemeConfig),
                ctx.spec.stallDtSteps, ctx.spec.stallProgress)
    contract = communication.localPhase('runner:setup', validate)
    # Domain tensors are already hashed by the neighbour builder. Avoid device
    # ordinals/pointers in cross-device metadata; ordinary scalar configs agree.
    communication.agree('runner:configuration', contract)
    communication.assertReplicated('runner:initial-state', _stateTensors(system),exact=True)
    if system.sourceTerms is not None:
        validateSourceClock(system, communication)
    return communication


def distributedStepStart(ctx, system, communication):
    original = ctx.config.dt
    def candidate():
        dtype = original.dtype if isinstance(original, torch.Tensor) else torch.float64
        if dtype not in (torch.float32, torch.float64):
            raise TypeError('Distributed timesteps require float32/float64 or a Python scalar.')
        dt = torch.as_tensor(original, dtype=dtype, device=communication.device).reshape(())
        if not math.isfinite(float(dt)) or float(dt) <= 0:
            raise ValueError('Distributed timestep must be positive and finite.')
        return dt
    local = communication.localPhase('runner:timestep-local', candidate)
    representation = (isinstance(original, torch.Tensor), str(local.dtype),
                      list(original.shape) if isinstance(original, torch.Tensor) else None)
    communication.agree('runner:timestep-representation', representation)
    # Retain the original tensor/scalar representation: converting a float32
    # device dt to Python float changes accumulated time and fresh stage times.
    reduced = communication.reduceScalar(local, operation='min', tag='runner:timestep')
    return reduced.to(original.device).reshape(original.shape) if isinstance(original, torch.Tensor) else float(reduced)


def distributedStepEnd(ctx, system, communication, step):
    def invalid():
        communication.validateAdjacency(system.adjacency)
        s = system.state
        return (not all(bool(torch.isfinite(t).all()) for t in _stateTensors(system).values())
                or any(bool((getattr(s,k) <= 0).any()) for k in ('densities','internalEnergies','pressures')))
    bad = communication.localPhase('runner:state-local', invalid)
    if communication.any(bad, tag='runner:invalid-state'):
        raise RuntimeError(f'Distributed accepted step {step} has a nonfinite or nonpositive state.')
    communication.assertReplicated('runner:accepted-state', _stateTensors(system),exact=True)
    communication.agree('runner:accepted-time', (step, float(system.t), system.adjacency.plan.signature))
    if system.sourceTerms is not None:
        validateSourceClock(system, communication)
