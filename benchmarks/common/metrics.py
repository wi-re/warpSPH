"""Error, order-of-convergence, and scaling-fit helpers shared by the suites.

All error measures are relative L2 on the full integrated field (every
particle). The comparison itself is promoted to float64 on CPU: the fields
are small (a few MB at benchmark resolutions) and the norm is a scalar, so
the promotion is free -- and it keeps the *measured* error from carrying
float32 norm noise at fine dt, where the temporal error being measured is
already near the float32 floor. The reference run shares the same spatial
discretization, so what these measure is temporal error only.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch


def relL2(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative L2 difference `||a - b|| / ||b||`, with `b` the reference.

    Returns `inf` when the reference norm is zero, `nan` when either field
    holds a non-finite value (diverged runs) -- the report layer renders
    both distinctly rather than plotting them as data.
    """
    a = a.detach().double()
    b = b.detach().double()
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        return float('nan')
    den = float(torch.linalg.norm(b))
    if den == 0.0:
        return float('inf')
    return float(torch.linalg.norm(a - b) / den)


def L1(a: torch.Tensor, b: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """Mean absolute difference `mean(|a - b|)` over the masked sample set.

    García-Senz & Cabezón (2026) Eq. (20), `L1 = (1/N_S) sum_{i in S} |p_i^SPH -
    p_i^ref|` -- deliberately NOT normalised, so every number quoted from that
    paper is comparable. Same convention as `relL2`: `b` is the reference,
    `nan` for a non-finite sample (diverged run), and `nan` for an empty mask
    (there is no sample set to average over).
    """
    a = a.detach().double()
    b = b.detach().double()
    if mask is not None:
        mask = mask.detach().bool()
        a, b = a[mask], b[mask]
    if a.numel() == 0 or not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        return float('nan')
    return float((a - b).abs().mean())


def effectiveOrder(errCoarse: float, errFine: float, ratio: float = 2.0) -> Optional[float]:
    """Measured convergence order between two errors at dt and dt/ratio.

    `None` when it cannot be defined (zero/negative/non-finite errors) -- a
    diverged run or a saturated reference should show up as "no order", not
    as a bogus number.
    """
    if (not (math.isfinite(errCoarse) and math.isfinite(errFine))
            or errCoarse <= 0.0 or errFine <= 0.0 or ratio <= 1.0):
        return None
    return math.log(errCoarse / errFine) / math.log(ratio)


def loglogFit(xs: Sequence[float], ys: Sequence[float]) -> Optional[Tuple[float, float, float]]:
    """Least-squares fit of `y = a * x**slope` in log-log space.

    Returns `(slope, a, r2)` over the positive points, or `None` when fewer
    than two positive points exist. Used for the runtime/memory scaling
    exponents in the performance suite (slope ~1 is linear-in-particles,
    ~2 is the naive all-pairs scaling).
    """
    pts = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y) and x > 0 and y > 0]
    if len(pts) < 2:
        return None
    lx = [math.log(x) for x, _ in pts]
    ly = [math.log(y) for _, y in pts]
    n = len(pts)
    mx = sum(lx) / n
    my = sum(ly) / n
    sxx = sum((x - mx) ** 2 for x in lx)
    if sxx == 0.0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(lx, ly)) / sxx
    a = math.exp(my - slope * mx)
    ssRes = sum((y - (my + slope * (x - mx))) ** 2 for x, y in zip(lx, ly))
    ssTot = sum((y - my) ** 2 for y in ly)
    r2 = 1.0 - ssRes / ssTot if ssTot > 0.0 else 1.0
    return slope, a, r2


def fmt(value, spec: str = '.4g', none: str = '-') -> str:
    """Format an optional/possibly-non-finite float for tables."""
    if value is None:
        return none
    if isinstance(value, float) and not math.isfinite(value):
        return 'nan' if math.isnan(value) else 'inf'
    return format(value, spec)


def mean(values: Sequence[float]) -> Optional[float]:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else None
