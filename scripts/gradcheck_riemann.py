#!/usr/bin/env python3
"""torch.autograd.gradcheck of the Riemann solvers (AV_PLAN Phase 7b.1, `modules/riemann`).

The solvers are pure `wp.func`s, so they are wrapped in a one-thread-per-state kernel and a `torch.autograd.Function`
(forward launch, backward through a `wp.Tape`). Checked against all six inputs `(rho, u, p)_L,R` for every
`RiemannSolver`'s `(p*, u*)` and for the HLLC flux, at states away from the kinks (the wave-pattern switches of HLLC /
Adaptive, the `max(., 0)` pressure clamp) and on both sides of the contact-speed sign of the flux.

    python scripts/gradcheck_riemann.py
"""

from __future__ import annotations

import os

os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import sys

import torch
import warp as wp

from warpSPHCore import scalar_t

from warpSPH.modules.riemann import RiemannSolver, hllcFlux, riemannStarState

GAMMA = 1.4
DEVICE = 'cpu'


@wp.kernel
def _star(solver: wp.int32, W: wp.array2d(dtype=scalar_t), gamma: scalar_t, out: wp.array2d(dtype=scalar_t)):
    k = wp.tid()
    p, u = riemannStarState(solver, W[k, 0], W[k, 1], W[k, 2], W[k, 3], W[k, 4], W[k, 5], gamma)
    out[k, 0] = p
    out[k, 1] = u


@wp.kernel
def _flux(W: wp.array2d(dtype=scalar_t), gamma: scalar_t, out: wp.array2d(dtype=scalar_t)):
    k = wp.tid()
    Fr, Fm, FE, S = hllcFlux(W[k, 0], W[k, 1], W[k, 2], W[k, 3], W[k, 4], W[k, 5], gamma)
    out[k, 0] = Fr
    out[k, 1] = Fm
    out[k, 2] = FE
    out[k, 3] = S


class _Launch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W, kernel, nOut, extra):
        Wwp = wp.from_torch(W.detach().contiguous(), dtype=scalar_t, requires_grad=True)
        out = wp.zeros((W.shape[0], nOut), dtype=scalar_t, requires_grad=True, device=DEVICE)
        tape = wp.Tape()
        with tape:
            wp.launch(kernel, dim=W.shape[0], inputs=[*extra[0], Wwp, scalar_t(GAMMA)], outputs=[out], device=DEVICE)
        ctx.tape, ctx.W, ctx.out = tape, Wwp, out
        return wp.to_torch(out).clone()

    @staticmethod
    def backward(ctx, gradOut):
        ctx.out.grad = wp.from_torch(gradOut.contiguous(), dtype=scalar_t)
        ctx.tape.backward()
        g = wp.to_torch(ctx.W.grad).clone()
        ctx.tape.zero()
        return g, None, None, None


def _states():
    """Smooth, well-separated states: compression, expansion, strong and weak, both contact-speed signs."""
    return torch.tensor([
        [1.0, 0.3, 1.0, 0.8, -0.2, 0.7],     # mild compression
        [1.0, -0.4, 1.0, 1.2, 0.5, 1.1],     # expansion
        [1.0, 0.1, 1.0, 0.125, 0.0, 0.1],    # Sod-like
        [0.9, 0.8, 2.0, 1.1, 0.7, 1.5],      # contact moving right
        [0.9, -0.8, 1.5, 1.1, -0.7, 2.0],    # contact moving left
        [1.0, 1.0, 1.0, 1.0, -1.0, 1.0],     # two shocks
    ], dtype=torch.float64)


def _check(label, fn) -> bool:
    W = _states().requires_grad_(True)
    print(f"\n=== {label}: torch.autograd.gradcheck ===")
    try:
        ok = torch.autograd.gradcheck(fn, (W,), eps=1e-6, atol=1e-5, rtol=1e-4)
        print("PASSED" if ok else "FAILED (gradcheck returned False)")
        return bool(ok)
    except Exception as exc:  # noqa: BLE001 - canary script
        print(f"FAILED: {type(exc).__name__}: {str(exc)[:600]}")
        return False


def main() -> int:
    wp.init()
    results = []
    for solver in RiemannSolver:
        results.append(_check(f"riemannStarState [{solver.name}]",
                              lambda W, s=solver: _Launch.apply(W, _star, 2, ([wp.int32(s.value)],))))
    results.append(_check("hllcFlux", lambda W: _Launch.apply(W, _flux, 4, ([],))))
    print(f"\n{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
