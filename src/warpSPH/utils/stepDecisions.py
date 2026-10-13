"""Host decisions a captured step defers until after its replay.

A step that is recorded into a CUDA graph cannot read a device value back (the one deferred pair-list check of
`systems/compSPH.py:finalize`, the learned rebuild decision). Inside `collectStepDecisions()` such a system registers a
`StepDecision` instead: the device `values` it would have read, and what to do with them on the host once they are
there. `utils/cudaGraph.py:GraphedIntegratorStep` reads every decision's values together with the step time (ONE transfer
per replay) and calls the decision back.

* `redo(host)`  -> True when the replayed step is not the step an eager run would have taken (a flag says a stage ran on a
  pair list that was too short): the replay is discarded and the step runs eagerly from the untouched input.
* `apply(host, system)` -> the step stands; the decision brings the host-side state forward (learn the buffer usage,
  rebuild the pair list for the next step in place, ...) and may change the returned `system`.
* `hostState()` -> a callable that puts the decision's host-side state back to what it is now (the first replay is
  compared with an eager step, which must not teach the host state twice).
"""

import contextlib
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch

__all__ = ['StepDecision', 'collectStepDecisions', 'recordingStepDecisions']


@dataclass
class StepDecision:
    values: torch.Tensor
    redo: Callable[[List[float]], bool]
    apply: Callable[[List[float], object], None]
    hostState: Optional[Callable[[], Callable[[], None]]] = None


_COLLECTING: List[Optional[list]] = [None]


@contextlib.contextmanager
def collectStepDecisions():
    previous = _COLLECTING[0]
    box: list = []
    _COLLECTING[0] = box
    try:
        yield box
    finally:
        _COLLECTING[0] = previous


def recordingStepDecisions() -> Optional[list]:
    """The list a deferred decision is appended to, or None when nothing is recording (read on the host as usual)."""
    return _COLLECTING[0]
