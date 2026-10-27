#!/usr/bin/env python
"""GIZMO (MFM / MFV) vs the exact Yee vortex on the IC files exported from our cases: same metrics as
`.tmp/ic_ab.py` (vortex KE / exact, L1 density, L1 velocity at t = 2)."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / ".tmp"))
import h5py
import numpy as np
import torch

import yee_valid

R = REPO / 'results' / 'gizmo_compare_2026-10-12'


class Dom:
    min = torch.tensor([-5.0, -5.0], dtype=torch.float64)
    max = torch.tensor([5.0, 5.0], dtype=torch.float64)


def analyse(path):
    with h5py.File(path, 'r') as h:
        t = float(h['Header'].attrs['Time'])
        p = h['PartType0']
        x = torch.tensor(p['Coordinates'][:, :2], dtype=torch.float64) - 5.0      # back to the [-5, 5] box
        v = torch.tensor(p['Velocities'][:, :2], dtype=torch.float64)
        rho = torch.tensor(p['Density'][:], dtype=torch.float64)
        m = torch.tensor(p['Masses'][:], dtype=torch.float64)
    rho_e, v_e, _ = yee_valid.exact(x, t, Dom, 0.0)
    ke = lambda vv: float((0.5 * m * (vv ** 2).sum(-1)).sum())
    return t, ke(v) / ke(v_e), float((rho - rho_e).abs().mean()), float((v - v_e).norm(dim=-1).mean())


for ic in ('yee_nx64_j0.0', 'yee_nx64_j0.0_cons', 'yee_nx64_j0.25', 'yee_nx64_j0.25_cons'):
    for v in ('mfm', 'mfv'):
        snaps = sorted((R / ic / v / 'output').glob('snapshot_*.hdf5'))
        t, ker, l1r, l1v = analyse(snaps[-1])
        print(f'[gizmo] {ic:22s} {v.upper()} | t={t:.2f} keR={ker:.4f} L1rho={l1r:.4e} L1v={l1v:.4e}', flush=True)
