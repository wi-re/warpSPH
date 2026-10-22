"""Lazy weakly-compressible acceleration/contact diagnostics."""
from collections.abc import Mapping
import torch
from warpSPHCore import compileGlue

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
    from ...utils.syncFree import quantilesFromSorted
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
        from ...utils.syncFree import deviceConstant
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


