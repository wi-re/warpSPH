"""`TimedBlock`, a CPU-(and optionally CUDA-event-)based profiling context
manager used by the commented-out timing scaffolds in `schemes/deltaSPH.py`
and `schemes/divergenceFree.py`. Only `cpu_ms` is actually populated on exit -- the
CUDA event is recorded but `cuda_ms` is left `None` here (its computation and
the `synchronize()` call it needs are commented out); the scaffolds that use
`use_cuda=True` compute `cuda_ms` themselves afterward from the block's
`_start_event`/`_end_event` rather than relying on `__exit__` for it.
"""

import time
import torch

__all__ = ['TimedBlock']


class TimedBlock:
    def __init__(self, name: str = "Timed block", use_cuda: bool = True, device=None):
        self.name = name
        self.device = device
        self.use_cuda = use_cuda and torch.cuda.is_available()

        self.cpu_ms = None
        self.cuda_ms = None

        self._cpu_start = None
        self._start_event = None
        self._end_event = None

    def __enter__(self):
        self._cpu_start = time.perf_counter()

        if self.use_cuda:
            # if self.device is not None:
            #     torch.cuda.synchronize(self.device)
            # else:
            #     torch.cuda.synchronize()

            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cpu_ms = (time.perf_counter() - self._cpu_start) * 1000.0

        if self.use_cuda:
            self._end_event.record()
            # self._end_event.synchronize()
            # self.cuda_ms = self._start_event.elapsed_time(self._end_event)
        #     print(f"[{self.name}] CPU: {self.cpu_ms:.3f} ms | CUDA: {self.cuda_ms:.3f} ms")
        # else:
        #     print(f"[{self.name}] CPU: {self.cpu_ms:.3f} ms | CUDA: N/A")

        return False