#!/usr/bin/env python
"""Export one of our case ICs (meshless state) to a GIZMO HDF5 IC file: the SAME particles, masses, velocities
and internal energies, shifted into [0, L)^2.

    python .tmp/gizmo_ic_export.py kh --nx 256 --out DIR
    python .tmp/gizmo_ic_export.py yee --nx 64 --jitter 0.25 [--consistent] --out DIR
    python .tmp/gizmo_ic_export.py gresho --nx 64 --jitter 0.0 --out DIR
"""
import argparse
import dataclasses
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / ".tmp"))
import h5py
import numpy as np
import torch

torch.set_grad_enabled(False)
from warpSPH.runner.runner import buildContext
from warpSPH.runner.caseSpec import CaseSpec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('case', choices=['kh', 'yee', 'gresho'])
    ap.add_argument('--nx', type=int, default=64)
    ap.add_argument('--jitter', type=float, default=0.0)
    ap.add_argument('--consistent', action='store_true', help='scheme-consistent masses (m = rho V_geo), as .tmp/ic_audit.py')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if a.case == 'kh':
        from warpSPH.cases.compressible.kelvinHelmholtz import kelvinHelmholtzCase as case
        params = dict(case.params, jitter=a.jitter)
    elif a.case == 'gresho':
        from warpSPH.cases.compressible.greshoVortex import greshoVortexCase as case
        params = dict(case.params, jitter=a.jitter)
    else:
        import yee_valid
        from warpSPH.cases.compressible.yeeVortex import yeeVortexCase
        yee_valid.OPT.update(jitter=a.jitter, bg=0.0)
        case = dataclasses.replace(yeeVortexCase, buildSystem=yee_valid.buildSystem)
        params = dict(case.params)
    spec = CaseSpec(caseName=case.name, scheme='MFM', params=params).merged(**case.defaults).merged(
        scheme='MFM', nx=a.nx, precision='float64', params=params)
    ctx = buildContext(case, spec)
    case.configureScheme(ctx)
    system = case.buildSystem(ctx)
    st, sc = system.state, ctx.schemeConfig
    if a.consistent:
        import ic_audit
        ic_audit.consistentIC(system, ctx.config, sc, st.densities.clone(), st.velocities.clone(), st.pressures.clone())
    dom = system.domain
    L = float((dom.max - dom.min)[0])
    pos = (st.positions - dom.min).double().cpu().numpy()
    N = pos.shape[0]
    q = np.zeros((N, 3))
    q[:, :2] = pos
    v = np.zeros((N, 3))
    v[:, :2] = st.velocities.double().cpu().numpy()
    m = st.masses.double().cpu().numpy()
    u = st.internalEnergies.double().cpu().numpy()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    name = f'{a.case}_nx{a.nx}_j{a.jitter}' + ('_cons' if a.consistent else '') + '_ics.hdf5'
    with h5py.File(Path(a.out) / name, 'w') as f:
        h = f.create_group('Header')
        npart = np.array([N, 0, 0, 0, 0, 0])
        h.attrs['NumPart_ThisFile'] = npart
        h.attrs['NumPart_Total'] = npart
        h.attrs['NumPart_Total_HighWord'] = 0 * npart
        h.attrs['MassTable'] = np.zeros(6)
        h.attrs['Time'] = 0.0
        h.attrs['NumFilesPerSnapshot'] = 1
        h.attrs['Flag_DoublePrecision'] = 1
        h.attrs['BoxSize'] = L
        p = f.create_group('PartType0')
        p.create_dataset('Coordinates', data=q)
        p.create_dataset('Velocities', data=v)
        p.create_dataset('ParticleIDs', data=np.arange(1, N + 1))
        p.create_dataset('Masses', data=m)
        p.create_dataset('InternalEnergy', data=u)
    print(f'{Path(a.out) / name}: N={N} L={L:g} gamma={float(sc.gamma):.4f} mass sum {m.sum():.6g} '
          f'u range {u.min():.4g}..{u.max():.4g} nngb(ours)={float(sc.nngb):.1f}', flush=True)


if __name__ == '__main__':
    main()
