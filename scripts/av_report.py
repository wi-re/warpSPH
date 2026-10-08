#!/usr/bin/env python3
"""The AV report (AV_PLAN.md Part 0): one table of metrics per (config, case).

    scripts/av_report.py --list
    scripts/av_report.py --config baseline --profile smoke
    scripts/av_report.py --config baseline --profile full           # video on
    scripts/av_report.py --config baseline --cases sod --repeat 2   # repro lock
    scripts/av_report.py --compare results/av_A results/av_B

A *config* is a named dissipation setup on a host scheme (`baseline` = Monaghan
with NoneSwitch / Cullen-Dehnen / Read-Hayfield). A *case* knows how to run itself
at each profile and how to turn the finished run into Part 0 metrics. Output goes
to `results/av_<stamp>/{results.json, report.md}` with the `meta` block of
`benchmarks/common/report.py` (versions + git SHAs).

Runs go one at a time, in this process (GPU runs are launch-bound; parallel ones
only slow each other down). `--profile full` records video and streams the
runner's progress rows; `smoke` (coarse, ~tens of steps) does neither -- it is a
plumbing check, not a physics run.

`--repeat N` runs every (config, case) N times and compares every scalar in the
records: the M0 reproducibility lock is `max relative difference < 1e-6`.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

try:  # `warpSPHBootstrap` resolves the sibling repos when run from the repo root
    import warpSPHBootstrap  # noqa: F401
except ImportError:
    pass

from benchmarks.common import report as rep                       # noqa: E402
from benchmarks.common.metrics import L1, fmt                      # noqa: E402

#: `tests/test_physics.py:_ENERGY_DRIFT` -- the existing per-scheme budget (Group E).
ENERGY_DRIFT = {'CompSPH': 1e-5, 'CRKSPH': 1e-4, 'Monaghan': 1e-4}
LOCK_TOL = 1e-6


# --------------------------------------------------------------------------- configs

@dataclass
class AVConfig:
    """A named dissipation setup: the host scheme and the switch, plus any case params."""
    name: str
    scheme: str = 'Monaghan'
    #: None: the scheme's own default switch (Monaghan: Rosswog 2020 since 2026-10-08)
    switch: Optional[str] = 'NoneSwitch'
    params: Dict[str, Any] = field(default_factory=dict)
    #: `ViscositySwitchConfig` fields (alpha_min, alpha_max, ...) set on the scheme config after
    #: the case configures it, so they apply to EVERY case. (As case `params` they only reached
    #: `sod`/`sodND`: `configureCompressible` ignores them, so through the first M0 baseline
    #: Read-Hayfield ran in its designed 0.2-1.0 alpha range on Sod only.)
    switchParams: Dict[str, Any] = field(default_factory=dict)
    #: `DiffusionParameters` fields overridden on the scheme config after the case
    #: configures it (e.g. `{'C_q': 2.0}`); these are scheme, not case, parameters.
    diffusion: Dict[str, Any] = field(default_factory=dict)
    #: scheme-config fields set after the case configures it, e.g. `{'owenTable': 'lattice'}`
    schemeFields: Dict[str, Any] = field(default_factory=dict)
    #: `SimulationConfig` fields set before the case builds its state (so the IC sees them too),
    #: e.g. `{'supportVolumeClamp': 'off'}` (OPEN_PROBLEMS §20); `--supportVolumeClamp` sets it for every config.
    simulation: Dict[str, Any] = field(default_factory=dict)
    #: Monaghan host: run on the pre-2026-10-08 defaults (`C_q = 0`, raw pair velocity, `LEGACY_MONAGHAN`) unless
    #: `diffusion` says otherwise -- every config written before the default changed means those, and the stored
    #: M0 references were taken with them. False: the scheme's current defaults (config `default`).
    legacyDefaults: bool = True


#: The Monaghan host's DiffusionParameters defaults before 2026-10-08 (now C_q = 2 coupled + Limited).
LEGACY_MONAGHAN = dict(C_q=0.0, velocityPairPolicy=0)


def effectiveDiffusion(cfg: 'AVConfig') -> Dict[str, Any]:
    """The DiffusionParameters overrides a config runs with (`legacyDefaults` applied)."""
    if cfg.scheme == 'Monaghan' and cfg.legacyDefaults:
        return {**LEGACY_MONAGHAN, **cfg.diffusion}
    return dict(cfg.diffusion)


#: R&H 2012 is designed with a narrower, higher-baseline alpha range
#: (`probe_cullen_sod.py`, `tests/test_shockCapturing.py`).
_RH = dict(alpha_min=0.2, alpha_max=1.0)          # applied through `switchParams`

CONFIGS: Dict[str, AVConfig] = {c.name: c for c in (
    # the Monaghan host's current defaults, whatever they are (2026-10-08: Rosswog 2020, alpha in [0, 1], C_q = 2
    # coupled, limited reconstruction -- user decision after the AV_PLAN sweep)
    AVConfig('default', switch=None, legacyDefaults=False),
    AVConfig('none', switch='NoneSwitch'),
    AVConfig('cullenDehnen2010', switch='CullenDehnen2010'),
    AVConfig('readHayfield2012', switch='ReadHayfield2012', switchParams=_RH),
    # C_q = 2 columns: the Monaghan default C_q = 0 has no quadratic term, so v_sig = C_l c
    # is zero in cold gas (Noh, Sedov's background) and nothing dissipates there.
    # AV_PLAN S3 switches (Monaghan host, C_q = 0): Balsara/Colagrossi are instantaneous
    # limiters (alpha = factor); MM97 / Rosswog2000 are source-and-decay with the paper's
    # alpha_inf = 0.1, C_1 = 0.2 and alpha_max = 2.
    AVConfig('balsara1995', switch='Balsara1995'),
    AVConfig('colagrossi2004', switch='Colagrossi2004'),
    AVConfig('morrisMonaghan1997', switch='MorrisMonaghan1997', switchParams=dict(alpha_min=0.1, alpha_max=2.0)),
    AVConfig('rosswog2000', switch='Rosswog2000', switchParams=dict(alpha_min=0.1, alpha_max=2.0)),
    # AV_PLAN Phase 2: Rosswog (2020) entropy trigger at the paper's alpha_0 = 0, alpha_max = 1
    # (eps_0 1e-4, eps_1 5e-2, decay 30 tau are the config defaults); the Q column for Sedov/Noh
    AVConfig('rosswog2020', switch='Rosswog2020', switchParams=dict(alpha_min=0.0, alpha_max=1.0)),
    AVConfig('rosswog2020Q', switch='Rosswog2020', switchParams=dict(alpha_min=0.0, alpha_max=1.0),
             diffusion=dict(C_q=2.0)),
    # CRKSPH host (AV_PLAN S4): its viscosity is the CRK limiter operator, not the pair operator
    AVConfig('crkNone', scheme='CRKSPH', switch='NoneSwitch'),
    AVConfig('crkCullenDehnen2010', scheme='CRKSPH', switch='CullenDehnen2010'),
    AVConfig('noneQ', switch='NoneSwitch', diffusion=dict(C_q=2.0)),
    AVConfig('cullenDehnen2010Q', switch='CullenDehnen2010', diffusion=dict(C_q=2.0)),
    AVConfig('readHayfield2012Q', switch='ReadHayfield2012', switchParams=_RH, diffusion=dict(C_q=2.0)),
    # AV_PLAN Phase 3: the reconstructed pair velocity on the M0 `none` column (alpha = 1, C_q = 0)
    AVConfig('noneLinear', switch='NoneSwitch', diffusion=dict(velocityPairPolicy=1)),
    AVConfig('noneLimited', switch='NoneSwitch', diffusion=dict(velocityPairPolicy=2)),
    # AV_PLAN Phase 5A (Chen & Nixon 2025): beta in {fixed 2, fixed 0.2, coupled 2 alpha} x {C&D, Rosswog 2020}
    *(AVConfig(f'cn{det}{tag}', switch=switch, switchParams=sp, diffusion=dict(C_q=cq, betaMode=mode))
      for det, switch, sp in (('CD', 'CullenDehnen2010', {}),
                              ('Rosswog', 'Rosswog2020', dict(alpha_min=0.0, alpha_max=1.0)))
      for tag, cq, mode in (('Fixed2', 2.0, 1), ('Fixed0p2', 0.2, 1), ('Coupled2', 2.0, 0))),
    # AV_PLAN Phase 5B: Sphenix (Borrow et al. 2022). Operator Eqs. (15)-(17), v_sig = c_i + c_j - 3 mu with the
    # whole term scaled by alpha_ij: the `Price2012` term (8) with C_l = 1, C_q = 3 coupled; Eq. (19)'s Balsara in the
    # pair coefficient; the plain (uncorrected) gradient for B, so no correction-matrix pass is added
    AVConfig('sphenix', switch='Sphenix2022', switchParams=dict(alpha_min=0.0, alpha_max=2.0),
             diffusion=dict(viscosityTerm=8, C_l=1.0, C_q=3.0, betaMode=0, balsaraPairLimiter=True,
                            correctReconstructionGradient=False)),
    # AV_PLAN Phase 6: Wadsley et al. (2017) Gasoline2 detector, alpha in [0, 2], beta = 2 fixed (their Eq. 17-18 operator
    # with pair means is the `Wadsley2008` term, 10)
    AVConfig('wadsley2017', switch='Wadsley2017', switchParams=dict(alpha_min=0.0, alpha_max=2.0),
             diffusion=dict(viscosityTerm=10, C_q=2.0, betaMode=1)),
    # AV_PLAN Phase 7 cross terms (rows 10-12): a detector with the limited reconstruction, beta = 2 fixed (row 12:
    # coupled, beta = 2 alpha)
    AVConfig('rosswogLimited', switch='Rosswog2020', switchParams=dict(alpha_min=0.0, alpha_max=1.0),
             diffusion=dict(C_q=2.0, betaMode=1, velocityPairPolicy=2)),
    AVConfig('wadsleyLimited', switch='Wadsley2017', switchParams=dict(alpha_min=0.0, alpha_max=2.0),
             diffusion=dict(viscosityTerm=10, C_q=2.0, betaMode=1, velocityPairPolicy=2)),
    AVConfig('rosswogLimitedCoupled', switch='Rosswog2020', switchParams=dict(alpha_min=0.0, alpha_max=1.0),
             diffusion=dict(C_q=2.0, betaMode=0, velocityPairPolicy=2)),
    # AV_PLAN Phase 5B robustness: cflFactor x 4 (0.3 -> 1.2)
    AVConfig('sphenixCfl4', switch='Sphenix2022', switchParams=dict(alpha_min=0.0, alpha_max=2.0),
             diffusion=dict(viscosityTerm=8, C_l=1.0, C_q=3.0, betaMode=0, balsaraPairLimiter=True,
                            correctReconstructionGradient=False), simulation=dict(cflFactor=1.2)),
    AVConfig('cullenDehnen2010Cfl4', switch='CullenDehnen2010', simulation=dict(cflFactor=1.2)),
    # AV_PLAN Phase 4: Garcia-Senz & Cabezon (2026) Table 1 rows 1-6. Their operator Eq. (6) is the `Price2012_98`
    # term (v_sig = alpha c_bar - beta w, pair means) with beta = 2 fixed; their switch is Read & Hayfield's at
    # alpha in [0.05, 1]. Rows 5/6: Balsara-modulated reconstruction, p = 1 / 2.
    *(AVConfig(f'gs{name}', switch=switch, switchParams=dict(alpha_min=0.05, alpha_max=1.0) if switch != 'NoneSwitch' else {},
               diffusion=dict(C_q=2.0, betaMode=1, velocityPairPolicy=policy, reconstructionBalsaraPower=p))
      for name, switch, policy, p in (
          ('AV', 'NoneSwitch', 0, 2.0),
          ('AVSW', 'ReadHayfield2012', 0, 2.0),
          ('AVSLR', 'NoneSwitch', 2, 2.0),
          ('AVSWSLR', 'ReadHayfield2012', 2, 2.0),
          ('AVSLRB', 'NoneSwitch', 3, 1.0),
          ('AVSLRB2', 'NoneSwitch', 3, 2.0))),
)}

#: Named groups, so `--config baseline` runs the three M0 columns.
GROUPS: Dict[str, List[str]] = {
    'baseline': ['none', 'cullenDehnen2010', 'readHayfield2012'],
    'baselineQ': ['noneQ', 'cullenDehnen2010Q', 'readHayfield2012Q'],
    'crk': ['crkNone', 'crkCullenDehnen2010'],
    'switchesS3': ['balsara1995', 'colagrossi2004', 'morrisMonaghan1997', 'rosswog2000'],
    'phase2': ['rosswog2020', 'rosswog2020Q'],
    'phase3': ['noneLinear', 'noneLimited'],
    'phase4': ['gsAV', 'gsAVSW', 'gsAVSLR', 'gsAVSWSLR', 'gsAVSLRB', 'gsAVSLRB2'],
    'phase5b': ['sphenix', 'cullenDehnen2010'],
    'phase6': ['wadsley2017', 'cullenDehnen2010'],
    'phase7cross': ['rosswogLimited', 'wadsleyLimited', 'rosswogLimitedCoupled'],
    'cfl4': ['sphenixCfl4', 'cullenDehnen2010Cfl4'],
    'phase5a': ['cnCDFixed2', 'cnCDFixed0p2', 'cnCDCoupled2', 'cnRosswogFixed2', 'cnRosswogFixed0p2', 'cnRosswogCoupled2'],
}


# --------------------------------------------------------------------------- run plumbing

@dataclass
class RunSpec:
    """What one case run needs: the case object, run kwargs, extra params."""
    case: Any
    kwargs: Dict[str, Any]
    params: Dict[str, Any] = field(default_factory=dict)


def _execute(spec: RunSpec, cfg: AVConfig, outDir: Path, tag: str, video: bool,
             avStride: int, extra: Optional[Callable] = None) -> Any:
    """Run one case under `cfg`; returns the `RunResult`, with the AV extras
    (`avReportDiagnostics`) recorded every `avStride` steps."""
    import dataclasses
    from warpSPH.runner import run
    from warpSPH.cases.compressible import avReportDiagnostics

    base = spec.case.diagnostics
    counter = [0]

    def diagnostics(ctx, state):
        row = base(ctx, state)
        if counter[0] % avStride == 0:
            row.update(avReportDiagnostics(ctx, state))
        if extra is not None:
            row.update(extra(ctx, state))
        counter[0] += 1
        return row

    configure = spec.case.configureScheme
    diffusion = effectiveDiffusion(cfg)

    def configureScheme(ctx):
        configure(ctx)
        for name, value in cfg.switchParams.items():
            setattr(ctx.schemeConfig.viscositySwitchParams, name, value)
        for name, value in diffusion.items():
            setattr(ctx.schemeConfig.diffusionParams, name, value)
        for name, value in cfg.simulation.items():
            setattr(ctx.config, name, value)
        for name, value in cfg.schemeFields.items():
            setattr(ctx.schemeConfig, name, value)

    case = dataclasses.replace(spec.case, diagnostics=diagnostics,
                               configureScheme=configureScheme if (diffusion or cfg.switchParams or cfg.simulation or cfg.schemeFields) else configure)
    kw = dict(spec.kwargs)
    kw.setdefault('scheme', cfg.scheme)
    params = dict(spec.params, viscositySwitch=cfg.switch, **cfg.params)
    if video:
        kw.update(plot=True, video=True, exportRoot=str(outDir / 'runs' / tag),
                  velocityAlarmPlotInterval=1)
    kw.update(progress=video, stallProgress=1e-3, quiet=True)
    return run(case, params=params, **kw)


def _trap(t: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapezoid(y, t)) if len(t) > 1 else float('nan')


def _common(res, cfg: AVConfig) -> Dict[str, Any]:
    """Metrics every case reports: Group B (alpha, AV energy), E (drift), F (cost)."""
    traj = res.trajectory
    out: Dict[str, Any] = {'diverged': bool(res.diverged), 'stopReason': res.stopReason,
                           'nSteps': int(res.nSteps),
                           'tFinal': float(res.state.t)}
    E = np.array([r['totalEnergy'] for r in traj])
    out['energyDrift'] = float(abs(E[-1] - E[0]) / abs(E[0]))
    out['energyDriftBudget'] = ENERGY_DRIFT.get(cfg.scheme, float('nan'))
    out['entropyGain'] = float(traj[-1]['entropy'] - traj[0]['entropy'])
    last = traj[-1]
    for k in ('alphaMean', 'alphaMax', 'alphaActiveFraction'):
        out[k] = float(last[k])
    rows = [r for r in traj if 'avPowerTotal' in r]
    if rows and all(math.isfinite(r['avPowerTotal']) for r in rows):
        t = np.array([r['t'] for r in rows])
        for k in ('Total', 'Linear', 'Quadratic'):
            out[f'avEnergy{k}'] = _trap(t, np.array([r[f'avPower{k}'] for r in rows]))
    # Chen & Nixon (2025) Eq. (6) quadratic/linear ratio (AV_PLAN Phase 5A): time means of the per-step median / p90
    cn = [r for r in traj if 'chenNixonRatioMedian' in r and math.isfinite(r['chenNixonRatioMedian'])]
    if cn:
        out['chenNixonRatioMedian'] = float(np.mean([r['chenNixonRatioMedian'] for r in cn]))
        out['chenNixonRatioP90'] = float(np.mean([r['chenNixonRatioP90'] for r in cn]))
    nb = [r for r in traj if 'neighboursMean' in r]
    if nb:
        out['neighboursMean'] = float(np.mean([r['neighboursMean'] for r in nb]))
        out['neighboursMax'] = float(max(r['neighboursMax'] for r in nb))
    out['wallMsPerStep'] = 1e3 * res.wallTime / max(res.nSteps, 1)   # coarse: setup included
    return out


# --------------------------------------------------------------------------- Sod (1D)

SOD_WINDOW = (0.0, 1.0, 0.5)           # exact-solution window [xl, xr], interface
SOD_L1_RANGE = (0.05, 0.868)           # García-Senz Eq. (20) window
SOD_IC = ((1.0, 1.0, 0.0), (0.1, 0.125, 0.0))   # (p, rho, u) left / right
GAMMA = 5 / 3


def _sodSpec(profile: str) -> RunSpec:
    from warpSPH.cases.sod import sodCase
    nx, nSteps = (100, 60) if profile == 'smoke' else (400, 400)
    return RunSpec(sodCase, dict(nx=nx, nSteps=nSteps), dict(_DA_SOD))


def _sodMetrics(res, cfg: AVConfig) -> Dict[str, Any]:
    from warpSPH.caseUtils.compressible.sod.sodSolution import solve
    out = _common(res, cfg)
    st = res.state.state
    t = float(res.state.t)
    x = st.positions[:, 0].detach().cpu().numpy()
    rho = st.densities.detach().cpu().numpy()
    P = st.pressures.detach().cpu().numpy()
    u = st.velocities[:, 0].detach().cpu().numpy()
    A = P / rho ** GAMMA
    pos, regions, vals = solve(SOD_IC[0], SOD_IC[1], SOD_WINDOW, t, gamma=GAMMA, npts=2000)

    m = (x >= SOD_L1_RANGE[0]) & (x <= SOD_L1_RANGE[1])
    out['L1_vx'] = L1(torch.tensor(u), torch.tensor(np.interp(x, vals['x'], vals['u'])),
                      torch.tensor(m))

    # contact spike: max P / A in a window around the exact contact, vs the exact R4 value
    cx = pos['Contact Discontinuity']
    sel = (vals['x'] > cx) & (vals['x'] < cx + 0.04)
    p4, rho4 = float(np.median(vals['p'][sel])), float(np.median(vals['rho'][sel]))
    w = (x > cx - 0.07) & (x < cx + 0.07)
    out['contactSpikeP'] = float(P[w].max() / p4 - 1.0)
    out['contactSpikeA'] = float(A[w].max() / (p4 / rho4 ** GAMMA) - 1.0)

    def regionErr(key, arr, lo, hi, ex):
        s = (x >= lo) & (x <= hi)
        return float(arr[s].mean() / ex - 1.0) if s.any() else float('nan')
    d = 0.03
    R3, R4 = regions['Region 3'], regions['Region 4']          # (p, rho, u)
    out['R3_rho_err'] = regionErr('rho', rho, pos['Foot of Rarefaction'] + d, cx - d, R3[1])
    out['R4_rho_err'] = regionErr('rho', rho, cx + d, pos['Shock'] - d, R4[1])
    out['R4_P_err'] = regionErr('P', P, cx + d, pos['Shock'] - d, R4[0])
    return out


# --------------------------------------------------------------------------- Gresho (2D)

def _greshoSpec(profile: str) -> RunSpec:
    from warpSPH.cases.greshoVortex import greshoVortexCase
    if profile == 'smoke':
        return RunSpec(greshoVortexCase, dict(nx=32, nSteps=40))
    return RunSpec(greshoVortexCase, dict(nx=100, tLimit=3.0))


def _shearingNohSpec(profile: str) -> RunSpec:
    from warpSPH.cases.shearingNoh import shearingNohCase
    if profile == 'smoke':
        return RunSpec(shearingNohCase, dict(nx=50, nSteps=40))
    return RunSpec(shearingNohCase, dict(nx=100, tLimit=0.6))


def _shearingNohMetrics(res, cfg: AVConfig) -> Dict[str, Any]:
    """AV_PLAN Phase 5A: the planar Noh front under a transverse shear. For gamma = 5/3 the exact front is at
    |x| = t/3 behind a 4x density jump (pre-shock rho = 1, planar); the front is the 95th percentile of |x| over the
    particles past the midpoint density 2.5."""
    out = _common(res, cfg)
    st = res.state.state
    t = float(res.state.t)
    x = np.abs(st.positions[:, 0].detach().cpu().numpy())
    rho = st.densities.detach().cpu().numpy()
    shocked = rho > 2.5
    out['shockFront'] = float(np.percentile(x[shocked], 95)) if shocked.any() else float('nan')
    out['shockFrontExact'] = t / 3.0
    out['shockFrontErr'] = out['shockFront'] / out['shockFrontExact'] - 1.0 if t > 0 else float('nan')
    return out


def _noh2dSpec(profile: str) -> RunSpec:
    from warpSPH.cases.noh import nohCase
    if profile == 'smoke':
        return RunSpec(nohCase, dict(dim=2, nx=50, nSteps=40))
    return RunSpec(nohCase, dict(dim=2, nx=100, tLimit=0.6))


def _noh2dMetrics(res, cfg: AVConfig) -> Dict[str, Any]:
    """AV_PLAN Phase 6: the cylindrical Noh implosion. Ahead of the shock the gas converges (div v = -1/r) without
    any radial velocity gradient: a divergence-based detector fires there, Wadsley's D (n radial, dv/dn = 0) must
    not. `preShockAlphaMean` is the mean alpha over 1.3 r_s < r < 0.35 (inside the box's periodic seam effects)."""
    from warpSPH.cases.noh import shockState
    out = _common(res, cfg)
    st = res.state.state
    t = float(res.state.t)
    vs = float(res.ctx.param('v_s'))
    r = st.positions.norm(dim=-1).detach().cpu().numpy()
    rho = st.densities.detach().cpu().numpy()
    alpha = st.alphas.detach().cpu().numpy() if getattr(st, 'alphas', None) is not None else np.ones_like(r)
    pre = (r > 1.3 * vs * t) & (r < 0.35)
    out['preShockAlphaMean'] = float(alpha[pre].mean()) if pre.any() else float('nan')
    rhoS, _ = shockState(res.ctx)
    post = r < 0.8 * vs * t
    out['postShockRhoErr'] = float(rho[post & (r > 0.2 * vs * t)].mean() / rhoS - 1.0) if post.any() else float('nan')
    out['postShockRhoExact'] = float(rhoS)
    return out


def _shearBoxSpec(profile: str) -> RunSpec:
    from warpSPH.cases.shearBox import shearBoxCase
    if profile == 'smoke':
        return RunSpec(shearBoxCase, dict(nx=32, nSteps=40))
    return RunSpec(shearBoxCase, dict(nx=64, tLimit=2.0))


def _shearBoxMetrics(res, cfg: AVConfig) -> Dict[str, Any]:
    """AV_PLAN Phase 5A: how much of the steady shear mode survives, and how the AV energy splits."""
    out = _common(res, cfg)
    amp = res.series('modeAmplitude')
    out['modeAmplitudeFinal'] = float(amp[-1])
    q, total = out.get('avEnergyQuadratic'), out.get('avEnergyTotal')
    if q is not None and total:
        out['avQuadraticFraction'] = float(q / total)
    return out


def _greshoExact(r: np.ndarray) -> np.ndarray:
    return np.where(r < 0.2, 5 * r, np.where(r < 0.4, 2 - 5 * r, 0.0))


def _greshoMetrics(res, cfg: AVConfig) -> Dict[str, Any]:
    out = _common(res, cfg)
    st = res.state.state
    pos = st.positions.detach().cpu().numpy()
    vel = st.velocities.detach().cpu().numpy()
    r = np.linalg.norm(pos, axis=1)
    vphi = (pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]) / np.maximum(r, 1e-12)
    m = r <= 0.5
    out['L1_vphi'] = L1(torch.tensor(vphi), torch.tensor(_greshoExact(r)), torch.tensor(m))
    speed = np.linalg.norm(vel, axis=1)
    speed0 = res.trajectory[0]
    out['peakSpeed'] = float(speed.max())
    out['angularMomentumLoss'] = float(1.0 - res.trajectory[-1]['angularMomentum']
                                       / res.trajectory[0]['angularMomentum'])
    return out


@dataclass
class CaseDef:
    name: str
    spec: Callable[[str], RunSpec]
    metrics: Callable[[Any, AVConfig], Dict[str, Any]]
    avStride: Callable[[str], int] = lambda profile: 1
    extra: Optional[Callable] = None        # per-step diagnostics beyond the common set


#: Every AV baseline run with the D&A Sod initial condition (the probes' and
#: `tests/test_shockCapturing.py`'s), not the sodND default `0.25 / 0.1795`.
_DA_SOD = dict(right_rho=0.125, right_pressure=0.1)


def _sodNDSpec(name: str):
    def spec(profile: str) -> RunSpec:
        from warpSPH.cases.sodND import sod2dCase, sod3dCase
        case = sod2dCase if name == 'sod2d' else sod3dCase
        small = dict(sod2d=(64, 40), sod3d=(28, 30))[name]
        kw = dict(nx=small[0], nSteps=small[1]) if profile == 'smoke' else (
            dict(nx=20) if name == 'sod3d' else {})
        # the D&A light state (rho 0.125) has larger supports than sodND's default
        # 0.25 one, so the periodic slab must be wider than its default 20 spacings
        return RunSpec(case, kw, dict(_DA_SOD, transverseSpacings=26))
    return spec


def _sedovSpec(profile: str) -> RunSpec:
    from warpSPH.cases.sedov import sedovCase
    kw = dict(dim=3, nx=14, nSteps=30) if profile == 'smoke' else dict(dim=3, nx=40)
    return RunSpec(sedovCase, kw)


def _sedovMetrics(res, cfg: AVConfig) -> Dict[str, Any]:
    out = _common(res, cfg)
    ctx = res.ctx
    st = res.state.state
    t = float(res.state.t)
    pos = st.positions.detach().cpu().numpy()
    rho = st.densities.detach().cpu().numpy()
    rho0 = float(ctx.param('rho0'))
    (vs, r2, v2, rho2, P2), _ = ctx.scratch['solution'].shockState(t), None
    r = np.linalg.norm(pos, axis=1)
    k = int(np.argmax(rho))
    out['peakRhoRatio'] = float(rho[k] / rho0)
    out['peakRhoExact'] = float(rho2 / rho0)
    out['peakOvershoots'] = bool(rho[k] / rho0 > rho2 / rho0)
    out['shockRadiusErr'] = float((r[k] - r2) / r2)
    out['E0Recovery'] = float(res.trajectory[0]['totalEnergy'] / float(ctx.param('E0')) - 1.0)
    return out


def _nohSpec(profile: str) -> RunSpec:
    from warpSPH.cases.noh import nohCase
    return RunSpec(nohCase, dict(nx=80, nSteps=100) if profile == 'smoke' else {})


def _nohMetrics(res, cfg: AVConfig) -> Dict[str, Any]:
    from warpSPH.cases.noh import shockState
    out = _common(res, cfg)
    ctx = res.ctx
    st = res.state.state
    t = float(res.state.t)
    x = np.abs(st.positions[:, 0].detach().cpu().numpy())
    rho = st.densities.detach().cpu().numpy()
    vs = float(ctx.param('v_s'))
    rhoS, _ = shockState(ctx)
    m = (x > 0.2 * vs * t) & (x < 0.8 * vs * t)          # away from the wall-heating core and the front
    out['postShockRhoErr'] = float(rho[m].mean() / rhoS - 1.0) if m.any() else float('nan')
    out['postShockRhoExact'] = float(rhoS)
    return out


def _yeeSpec(profile: str) -> RunSpec:
    from warpSPH.cases.yeeVortex import yeeVortexCase
    return RunSpec(yeeVortexCase, dict(nSteps=30) if profile == 'smoke' else dict(tLimit=4.0))


def _yeeMetrics(res, cfg: AVConfig) -> Dict[str, Any]:
    out = _common(res, cfg)
    p = res.ctx.spec.params
    beta, xc, yc = float(p['beta']), float(p['xc']), float(p['yc'])
    st = res.state.state
    pos = st.positions.detach().cpu().numpy() - np.array([xc, yc])
    vel = st.velocities.detach().cpu().numpy()
    r = np.linalg.norm(pos, axis=1)
    exact = beta / (2 * np.pi) * np.exp((1 - r ** 2) / 2) * r      # |v|, stationary vortex
    m = r <= 3.0                                                     # inside the buffer rings
    out['L1_speed'] = L1(torch.tensor(np.linalg.norm(vel, axis=1)), torch.tensor(exact), torch.tensor(m))
    out['peakSpeedRatio'] = float(np.linalg.norm(vel, axis=1)[m].max() / (beta / (2 * np.pi)))
    out['angularMomentumLoss'] = float(1.0 - res.trajectory[-1]['angularMomentum']
                                       / res.trajectory[0]['angularMomentum'])
    return out


def _waveSpec(profile: str) -> RunSpec:
    from warpSPH.cases.linearWave import linearWaveCase
    # A = 1e-4, not the case's 1e-6: at 1e-6 in float32 the velocity noise is ~7x the
    # signal. tLimit = 1/4 crossing is the standing wave's maximum (a full crossing, the
    # case default, has an analytic velocity of zero).
    params = dict(A=1e-4)
    return RunSpec(linearWaveCase, dict(nx=100, nSteps=100) if profile == 'smoke'
                   else dict(tLimit=0.25), params)


def _waveMetrics(res, cfg: AVConfig) -> Dict[str, Any]:
    """Compare against the analytic standing wave. An initial pressure sine
    `dP = A sin(kx)` at rest splits into two travelling waves, i.e.
    `v(x, t) = -(A / (rho0 c_s)) cos(kx) sin(k c_s t)`; `waveVelocityErr` is the
    projected-coefficient error in units of `A / (rho0 c_s)`, `waveResidualRms` the
    rms of what is *not* that mode (the noise floor -- A = 1e-6 in float32 is small)."""
    out = _common(res, cfg)
    p = res.ctx.spec.params
    k = 2 * np.pi / float(p['lamda'])
    c, A, rho0 = float(p['c_s']), float(p['A']), float(p['rho0'])
    t = float(res.state.t)
    st = res.state.state
    x = st.positions[:, 0].detach().cpu().numpy()
    v = st.velocities[:, 0].detach().cpu().numpy()
    unit = A / (rho0 * c)
    coef = 2.0 / len(x) * (v * np.cos(k * x)).sum()
    out['waveVelocityErr'] = float(abs(coef - (-unit * np.sin(k * c * t))) / unit)
    out['waveVelocityCoefficient'] = float(coef / unit)
    out['waveExactCoefficient'] = float(-np.sin(k * c * t))
    out['waveResidualRms'] = float(np.sqrt(np.mean((v - coef * np.cos(k * x)) ** 2)) / unit)
    return out


def _khMode(ctx, state) -> Dict[str, float]:
    """McNally et al. (2012) Eq. (6)-(8) transverse-velocity mode amplitude."""
    st = state.state
    k = float(ctx.param('freq')) * np.pi
    x, y = st.positions[:, 0], st.positions[:, 1]
    vy = st.velocities[:, 1]
    yy = torch.where(y < 0.5, y, 1.0 - y)
    d = torch.exp(-k * (yy - 0.25).abs())
    s = (vy * torch.sin(k * x) * d).sum()
    c = (vy * torch.cos(k * x) * d).sum()
    return {'khAmplitude': (2.0 * torch.sqrt(s * s + c * c) / d.sum()).item()}


def _khSpec(profile: str) -> RunSpec:
    from warpSPH.cases.kelvinHelmholtz import kelvinHelmholtzCase
    if profile == 'smoke':
        return RunSpec(kelvinHelmholtzCase, dict(nx=48, nSteps=30))
    return RunSpec(kelvinHelmholtzCase, dict(nx=128, tLimit=1.5))


def _khMetrics(res, cfg: AVConfig) -> Dict[str, Any]:
    out = _common(res, cfg)
    amp = res.series('khAmplitude')
    t = res.series('t')
    out['khAmplitude0'] = float(amp[0])
    out['khAmplitudeFinal'] = float(amp[-1])
    out['khAmplitudeMax'] = float(amp.max())
    i = int(np.argmin(np.abs(t - 1.5)))
    out['khAmplitudeAt1p5'] = float(amp[i]) if abs(t[i] - 1.5) < 0.05 else float('nan')
    out['khReference1p5'] = 14.79e-2          # McNally et al. (2012)
    return out


def _rtSpec(profile: str) -> RunSpec:
    from warpSPH.cases.rayleighTaylor import rayleighTaylorCase
    if profile == 'smoke':
        return RunSpec(rayleighTaylorCase, dict(nx=32, nSteps=30))
    return RunSpec(rayleighTaylorCase, dict(nx=64, tLimit=4.0))


def _rtMetrics(res, cfg: AVConfig) -> Dict[str, Any]:
    """Interface tips (reported, no reference): the lowest heavy-fluid and highest
    light-fluid particle, split at the mean of the two initial densities."""
    out = _common(res, cfg)
    p = res.ctx.spec.params
    mid = 0.5 * (float(p['rho_low']) + float(p['rho_high']))
    st = res.state.state
    fluid = (st.kinds == 0).detach().cpu().numpy()
    y = st.positions[:, 1].detach().cpu().numpy()
    rho = st.densities.detach().cpu().numpy()
    heavy, light = fluid & (rho > mid), fluid & (rho <= mid)
    out['heavyMinY'] = float(y[heavy].min()) if heavy.any() else float('nan')
    out['lightMaxY'] = float(y[light].max()) if light.any() else float('nan')
    out['mixingWidth'] = out['lightMaxY'] - out['heavyMinY']
    out['maxVelocity'] = float(res.series('maxVelocity').max())
    return out


CASES: Dict[str, CaseDef] = {c.name: c for c in (
    CaseDef('sod', _sodSpec, _sodMetrics),
    CaseDef('sod2d', _sodNDSpec('sod2d'), _sodMetrics),
    CaseDef('sod3d', _sodNDSpec('sod3d'), _sodMetrics, lambda p: 1 if p == 'smoke' else 5),
    CaseDef('sedov', _sedovSpec, _sedovMetrics, lambda p: 1 if p == 'smoke' else 10),
    CaseDef('noh', _nohSpec, _nohMetrics),
    CaseDef('gresho', _greshoSpec, _greshoMetrics, lambda p: 1 if p == 'smoke' else 5),
    CaseDef('yee', _yeeSpec, _yeeMetrics, lambda p: 1 if p == 'smoke' else 10),
    CaseDef('linearWave', _waveSpec, _waveMetrics, lambda p: 1 if p == 'smoke' else 10),
    CaseDef('kelvinHelmholtz', _khSpec, _khMetrics, lambda p: 1 if p == 'smoke' else 5, _khMode),
    CaseDef('rayleighTaylor', _rtSpec, _rtMetrics, lambda p: 1 if p == 'smoke' else 5),
    # AV_PLAN Phase 5A; not in the default case list (--cases shearBox), so older --compare runs stay comparable
    CaseDef('shearBox', _shearBoxSpec, _shearBoxMetrics, lambda p: 1 if p == 'smoke' else 5),
    CaseDef('shearingNoh', _shearingNohSpec, _shearingNohMetrics, lambda p: 1 if p == 'smoke' else 5),
    CaseDef('noh2d', _noh2dSpec, _noh2dMetrics, lambda p: 1 if p == 'smoke' else 5),
)}

#: Cases a run without `--cases` covers (the M0 ten).
DEFAULT_CASES = ['sod', 'sod2d', 'sod3d', 'sedov', 'noh', 'gresho', 'yee', 'linearWave', 'kelvinHelmholtz',
                 'rayleighTaylor']


# --------------------------------------------------------------------------- driver

def runOne(cfg: AVConfig, case: CaseDef, profile: str, outDir: Path, video: bool,
           tag: str, runParams: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    spec = case.spec(profile)
    if runParams:
        spec = dataclasses.replace(spec, kwargs=dict(spec.kwargs, **runParams))
    started = time.perf_counter()
    res = _execute(spec, cfg, outDir, tag, video, case.avStride(profile), case.extra)
    try:
        metrics = case.metrics(res, cfg)
    except Exception as exc:          # noqa: BLE001
        # a diverged run (e.g. the cfl x 4 robustness configs) can leave a state the case metrics cannot read
        # (empty selections, NaN positions): record the divergence instead of losing the rest of the job
        if not res.diverged:
            raise
        metrics = dict(diverged=True, stopReason=res.stopReason, nSteps=int(res.nSteps), tFinal=float(res.state.t),
                       metricsError=f'{type(exc).__name__}: {exc}')
    metrics['videoPath'] = res.videoPath
    return dict(config=cfg.name, case=case.name, profile=profile, scheme=cfg.scheme,
                switch=cfg.switch, seconds=time.perf_counter() - started, metrics=metrics)


def _scalars(rec: Dict[str, Any]) -> Dict[str, float]:
    return {k: v for k, v in rec['metrics'].items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
            and k not in ('wallMsPerStep',)}          # timing is not part of the lock


def lockCheck(reps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Max relative difference over every scalar between repeated runs."""
    worst, where = 0.0, None
    a = _scalars(reps[0])
    for other in reps[1:]:
        b = _scalars(other)
        for k, va in a.items():
            vb = b.get(k)
            if vb is None or (math.isnan(va) and math.isnan(vb)):
                continue
            rel = abs(va - vb) / max(abs(va), abs(vb), 1e-300) if va != vb else 0.0
            if rel > worst:
                worst, where = rel, k
    return dict(maxRelDiff=worst, worstMetric=where, locked=worst < LOCK_TOL)


_SOD_COLS = ['L1_vx', 'contactSpikeP', 'contactSpikeA', 'R3_rho_err', 'R4_rho_err', 'R4_P_err']
COLUMNS = {
    'sod': _SOD_COLS, 'sod2d': _SOD_COLS, 'sod3d': _SOD_COLS,
    'gresho': ['L1_vphi', 'peakSpeed', 'angularMomentumLoss'],
    'sedov': ['peakRhoRatio', 'peakRhoExact', 'peakOvershoots', 'shockRadiusErr', 'E0Recovery'],
    'noh': ['postShockRhoErr', 'postShockRhoExact'],
    'yee': ['L1_speed', 'peakSpeedRatio', 'angularMomentumLoss'],
    'linearWave': ['waveVelocityErr', 'waveVelocityCoefficient', 'waveExactCoefficient', 'waveResidualRms'],
    'kelvinHelmholtz': ['khAmplitude0', 'khAmplitudeAt1p5', 'khAmplitudeMax', 'khReference1p5'],
    'rayleighTaylor': ['heavyMinY', 'lightMaxY', 'mixingWidth', 'maxVelocity'],
    'shearBox': ['modeAmplitudeFinal', 'avQuadraticFraction'],
    'shearingNoh': ['shockFront', 'shockFrontExact', 'shockFrontErr'],
    'noh2d': ['preShockAlphaMean', 'postShockRhoErr', 'postShockRhoExact'],
}
COMMON_COLUMNS = ['alphaMean', 'alphaMax', 'alphaActiveFraction', 'avEnergyLinear',
                  'avEnergyQuadratic', 'chenNixonRatioMedian', 'entropyGain', 'energyDrift', 'neighboursMean',
                  'wallMsPerStep']


def renderMarkdown(records: List[Dict[str, Any]], locks: Dict[str, Any], meta: Dict[str, Any],
                   profile: str) -> List[tuple]:
    sections = []
    for case in dict.fromkeys(r['case'] for r in records):
        cols = COLUMNS.get(case, []) + COMMON_COLUMNS
        rows = []
        for r in (x for x in records if x['case'] == case):
            m = r['metrics']
            flag = ' **DIVERGED**' if m.get('diverged') else ''
            rows.append([r['config'] + flag] + [fmt(m.get(c), '.4g') for c in cols])
        sections.append((f'{case} ({profile})', rep.mdTable(['config'] + cols, rows), []))
        # a diverged run recorded without its metrics (`runOne`) has no drift to check
        budgets = [(r['config'], r['metrics']['energyDrift'], r['metrics']['energyDriftBudget'])
                   for r in records if r['case'] == case and 'energyDrift' in r['metrics']]
        over = [f'{c} ({d:.2e} > {b:.0e})' for c, d, b in budgets if d > b]
        sections.append((f'{case}: energy-drift budget (Group E)',
                         'all within budget' if not over else 'OVER: ' + ', '.join(over), []))
    if locks:
        rows = [[k, f"{v['maxRelDiff']:.2e}", v['worstMetric'] or '-',
                 'LOCKED' if v['locked'] else 'NOT LOCKED'] for k, v in locks.items()]
        sections.append((f'Reproducibility lock (tol {LOCK_TOL:g})',
                         rep.mdTable(['run', 'max rel diff', 'worst metric', 'state'], rows), []))
    sections.append(('meta', '```\n' + '\n'.join(f'{k}: {v}' for k, v in meta.items()) + '\n```', []))
    return sections


#: Not physics: timing and bookkeeping never take part in a comparison.
_COMPARE_SKIP = {'wallMsPerStep', 'stopReason'}


def compareReports(a: Path, b: Path, tol: float) -> int:
    """Per-scalar relative difference between two report runs, matched on
    (config, case). `0` if every shared scalar agrees to `tol` (0 = bit-identical)
    and the two runs cover the same (config, case) pairs, else `1`. This is the
    instrument for AV_PLAN's "bit-for-bit vs M0" refactor checks."""
    import json

    def load(path: Path) -> Dict[tuple, Dict[str, Any]]:
        path = Path(path)
        path = path / 'results.json' if path.is_dir() else path
        return {(r['config'], r['case']): r['metrics'] for r in json.loads(path.read_text())['records']}

    A, B = load(a), load(b)
    bad = 0
    only = sorted(set(A) ^ set(B))
    if only:
        print(f'pairs present in only one report: {only}')
        bad += 1
    rows = []
    for key in sorted(set(A) & set(B)):
        ma, mb = A[key], B[key]
        worst, where, nShared = 0.0, None, 0
        for k in sorted(set(ma) & set(mb)):
            va, vb = ma[k], mb[k]
            if k in _COMPARE_SKIP or isinstance(va, bool) or not isinstance(va, (int, float)) \
                    or not isinstance(vb, (int, float)):
                continue
            nShared += 1
            if va == vb or (math.isnan(va) and math.isnan(vb)):
                continue
            rel = abs(va - vb) / max(abs(va), abs(vb), 1e-300)
            if rel > worst:
                worst, where = rel, k
        # a metric the reference (A) has and B lost fails; one B added (a later report's new row, e.g. the Phase 5A
        # Chen & Nixon ratio against an M0 reference) is listed but does not
        missing = sorted((set(ma) - set(mb)) - _COMPARE_SKIP)
        added = sorted((set(mb) - set(ma)) - _COMPARE_SKIP)
        ok = worst <= tol and not missing
        bad += 0 if ok else 1
        rows.append([f'{key[0]}/{key[1]}', nShared, f'{worst:.2e}', where or '-',
                     ','.join(missing + [f'+{k}' for k in added]) or '-', 'OK' if ok else 'DIFF'])
    print(rep.mdTable(['pair', 'scalars', 'max rel diff', 'worst metric', 'keys in one only', 'state'], rows))
    print(f'\n{"IDENTICAL" if not bad and tol == 0 else ("WITHIN TOL" if not bad else "DIFFERENT")}'
          f' (tol {tol:g}): {len(rows)} pairs compared, {bad} differing')
    return 1 if bad else 0


def mergeReports(dirs: List[Path], out: Path) -> int:
    """Combine report directories into one: records are matched on (config, case) and a
    later directory replaces an earlier one; `locks.json` files are merged the same way.
    Writes `results.json` + `report.md` into `out` (meta = the last directory's)."""
    import json
    records: Dict[tuple, Dict[str, Any]] = {}
    locks: Dict[str, Any] = {}
    meta: Dict[str, Any] = {}
    for d in dirs:
        d = Path(d)
        payload = json.loads((d / 'results.json').read_text())
        meta = dict(payload['meta'], mergedFrom=[str(x) for x in dirs])
        for r in payload['records']:
            records[(r['config'], r['case'])] = r
        if (d / 'locks.json').exists():
            locks.update(json.loads((d / 'locks.json').read_text()))
    out.mkdir(parents=True, exist_ok=True)
    recs = list(records.values())
    # drop lock entries for pairs that no longer exist
    locks = {k: v for k, v in locks.items() if tuple(k.split('/')) in records}
    rep.writeResults(out, 'av', meta, recs)
    profile = meta.get('profile', 'full')
    rep.writeSummary(out, f'AV report ({profile}, merged)', renderMarkdown(recs, locks, meta, profile))
    (out / 'summary.md').rename(out / 'report.md')
    (out / 'locks.json').write_text(json.dumps(locks, indent=2))
    print(f'[av_report] merged {len(dirs)} reports, {len(recs)} pairs -> {out}/report.md')
    return 0


def _parseKV(pairs: List[str]) -> Dict[str, Any]:
    """`KEY=VALUE` -> dict, VALUE as a Python literal when it is one (True, 0.5), else the string."""
    import ast
    out = {}
    for kv in pairs:
        k, v = kv.split('=', 1)
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--config', default='baseline', help='config or group name (see --list)')
    ap.add_argument('--profile', choices=('smoke', 'full'), default='smoke')
    ap.add_argument('--cases', nargs='*', default=None)
    ap.add_argument('--repeat', type=int, default=1, help='runs per (config, case); >1 checks the lock')
    ap.add_argument('--video', dest='video', action='store_true', default=None)
    ap.add_argument('--noVideo', dest='video', action='store_false')
    ap.add_argument('--out', default=None)
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--compare', nargs=2, metavar=('A', 'B'),
                    help='compare two report dirs / results.json (exit 1 if they differ)')
    ap.add_argument('--merge', nargs='+', metavar='DIR',
                    help='merge report dirs (later overrides earlier per config/case) into --out')
    ap.add_argument('--caseParam', nargs='*', default=[], metavar='KEY=VALUE',
                    help="case params for every config, e.g. adaptiveSupportScheme=Monaghan (SUPPORT_SOLVER_PLAN)")
    ap.add_argument('--schemeParam', nargs='*', default=[], metavar='KEY=VALUE',
                    help="scheme-config fields for every config, e.g. owenTable=lattice (SUPPORT_SOLVER_PLAN)")
    ap.add_argument('--runParam', nargs='*', default=[], metavar='KEY=VALUE',
                    help="run() keyword overrides for every case, e.g. nx=256 tLimit=3.0 (AV_PLAN sweep: KH at nx 256)")
    ap.add_argument('--supportVolumeClamp', choices=('always', 'walls', 'off'), default=None,
                    help="set SimulationConfig.supportVolumeClamp for every config (OPEN_PROBLEMS §20)")
    ap.add_argument('--tol', type=float, default=0.0,
                    help='--compare tolerance on the relative difference (0 = bit-identical)')
    args = ap.parse_args()

    if args.merge:
        return mergeReports([Path(d) for d in args.merge], rep.outDirFor('av', args.out))

    if args.compare:
        return compareReports(Path(args.compare[0]), Path(args.compare[1]), args.tol)

    if args.list:
        print('configs:', ', '.join(CONFIGS), '\ngroups: ',
              ', '.join(f'{g}={v}' for g, v in GROUPS.items()), '\ncases:  ', ', '.join(CASES))
        return 0

    names = GROUPS.get(args.config, [args.config])
    cases = [CASES[c] for c in (args.cases or DEFAULT_CASES)]
    video = (args.profile == 'full') if args.video is None else args.video
    outDir = rep.outDirFor('av', args.out)
    meta = rep.environmentMeta(extra=dict(profile=args.profile, config=args.config, repeat=args.repeat,
                                          video=video, supportVolumeClamp=args.supportVolumeClamp,
                                          caseParam=args.caseParam, schemeParam=args.schemeParam,
                                          runParam=args.runParam))
    print(f'[av_report] {names} x {[c.name for c in cases]}  profile={args.profile} '
          f'repeat={args.repeat} video={video}\n[av_report] -> {outDir}', flush=True)

    records: List[Dict[str, Any]] = []
    locks: Dict[str, Any] = {}
    for name in names:
        for case in cases:
            reps = []
            for i in range(args.repeat):
                print(f'[av_report] {name} / {case.name}  (run {i + 1}/{args.repeat})', flush=True)
                cfg = CONFIGS[name]
                if args.caseParam:
                    cfg = dataclasses.replace(cfg, params=dict(cfg.params, **_parseKV(args.caseParam)))
                if args.schemeParam:
                    cfg = dataclasses.replace(cfg, schemeFields=dict(cfg.schemeFields, **_parseKV(args.schemeParam)))
                if args.supportVolumeClamp is not None:
                    cfg = dataclasses.replace(cfg, simulation=dict(cfg.simulation, supportVolumeClamp=args.supportVolumeClamp))
                reps.append(runOne(cfg, case, args.profile, outDir,
                                   video and i == 0, f'{name}_{case.name}', _parseKV(args.runParam)))
            records.append(reps[0])
            if args.repeat > 1:
                locks[f'{name}/{case.name}'] = lockCheck(reps)

    rep.writeResults(outDir, 'av', meta, records)
    rep.writeSummary(outDir, f'AV report ({args.profile})', renderMarkdown(records, locks, meta, args.profile))
    (outDir / 'summary.md').rename(outDir / 'report.md')
    if locks:
        (outDir / 'locks.json').write_text(__import__('json').dumps(locks, indent=2))
    print(f'[av_report] wrote {outDir}/report.md')
    return 0 if all(v['locked'] for v in locks.values()) else (0 if not locks else 2)


if __name__ == '__main__':
    sys.exit(main())
