"""Failure-only recovery of an accepted eager chunked driven-flow boundary.

The particle state is retained by the integrator, not cloned every step. Only
small mutable forcing/clock metadata is saved before the attempt. This is not
recovery from an asynchronous device fault or a partially accepted hook.
"""
from dataclasses import replace
from pathlib import Path
import traceback

import torch


class RunAllocationError(MemoryError):
    """An integrator allocation failed; ``result`` retains the accepted system.

    ``checkpointPath`` names an atomically published native restart, or is None.
    ``recoveryError`` explains a failed recovery without hiding the original
    allocation exception (``__cause__``). No failed trial is an accepted step.
    """
    def __init__(self, original, result, checkpointPath=None, recoveryError=None):
        super().__init__(str(original))
        self.result = result
        self.checkpointPath = checkpointPath
        self.recoveryError = recoveryError


def captureBoundary(ctx, system):
    from ..io.chunkCheckpoint import _scope
    token = ctx.scratch.get('_rowChunkAccepted')
    if token is None or token[0]() is not system:
        return None
    try:
        _scope(ctx, system)
    except (NotImplementedError, ValueError):
        return None
    return (token[1], system.sourceTerms.acceleration.state_dict(), ctx.config.dt)


def recoverAllocation(ctx, result, boundary, original):
    """Best effort: preserve the primary error even if sync/save also fails."""
    # Keep the diagnostic text while releasing failed stage frames and their
    # potentially enormous partial pair buffers before attempting host IO.
    originalText = ''.join(traceback.format_exception(original))
    traceback.clear_frames(original.__traceback__)
    result.stopReason = 'memory allocation failed'
    result.diverged = True
    path, recoveryError = None, None
    if boundary is not None:
        result.nSteps = boundary[0] - ctx.spec.resumeStepOffset
    try:
        if boundary is None:
            raise RuntimeError('No qualified accepted chunked checkpoint boundary is available.')
        completed, forcing, nextDt = boundary
        if torch.device(ctx.device).type == 'cuda':
            torch.cuda.synchronize(ctx.device)
        from ..modules.forcing import FourierOU
        from ..io.chunkCheckpoint import markCheckpointBoundary, writeChunkedCheckpoint
        # initialize() replaces pairWork on the input system. Its trial scratch
        # must not be treated as part of the accepted particle state.
        scratch = {name: None for name in ('pairWork', 'rerunRequest', 'supportGrid', 'listGrid', 'meshlessStages')
                   if hasattr(result.state, name)}
        accepted = replace(result.state, **scratch, sourceTerms=replace(result.state.sourceTerms,
                           acceleration=FourierOU.from_state_dict(forcing)))
        if hasattr(result.state, 'pairWork'):
            result.state.pairWork = None
        result.state = accepted
        result.nSteps = completed - ctx.spec.resumeStepOffset
        ctx.scratch['driving'] = accepted.sourceTerms.acceleration
        ctx.scratch['completedSteps'] = completed
        ctx.config.dt = nextDt
        markCheckpointBoundary(ctx, accepted, completed)
        if ctx.exportPath is None:
            # Distributed proof-of-concept runs have automatic output disabled.
            # A deterministic path lets every rank enter the same atomic native
            # write without a new transport operation or timestamp race.
            from ..io.export import resolveExportRoot
            ctx.exportPath = str(Path(resolveExportRoot(ctx.spec.exportRoot)) /
                f'{ctx.spec.caseName}_allocationFailure_step{completed}_{float(accepted.t).hex()}')
            result.exportPath = ctx.exportPath
        path = writeChunkedCheckpoint(Path(ctx.exportPath)/'allocationFailure.h5', ctx, accepted)
    except Exception as secondary:
        recoveryError = f'{type(secondary).__name__}: {secondary}'
    error = RunAllocationError(original, result, path, recoveryError)
    error.add_note('Original allocation failure:\n' + originalText)
    error.add_note('Accepted-state checkpoint: ' + (path or 'not published; ' + recoveryError))
    return error
