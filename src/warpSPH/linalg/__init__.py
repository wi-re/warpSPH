"""Small dense linear algebra on the GPU: batched ops on `(n, d, d)` tensors (d <= 3) that torch only offers through host-synchronising
LAPACK / cuSOLVER calls (`torch.linalg.pinv` reads the host to check its info flag), as warp kernels, plus the `@wp.func`s the same ops are
built from so other kernels can call them inline.

* `pinv.py` -- the symmetric pseudo-inverse (`pinvSymmetric`, `applyPinv`, and `symmetricPinv3` for use inside a kernel).

Linear solves (`solveSymmetric`, small dense systems for the implicit shifting and Newton blocks) belong here too, when they are written; the
rule for what goes in: batched, sync-free, forward-checked against the float64 torch result, with the torch call as the fallback for inputs
that need an autograd tape.
"""

from .pinv import applyPinv, loadPadded3x3, pinvSymmetric, symmetricPinv3

__all__ = ['applyPinv', 'pinvSymmetric', 'symmetricPinv3', 'loadPadded3x3']
