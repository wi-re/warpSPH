"""`WeaklyCompressibleSPHConfig.timeCentredContinuity` (CEILING_STICKING_PLAN.md §3).

symplecticEuler advances (rho, v) by the explicit midpoint rule, which grows
every acoustic mode by |lambda|^2 = 1 + (omega dt)^4/4 per step; with the
continuity density advanced by the positions' time-centred velocity the pair
is Stormer-Verlet and conserves the acoustic energy. Checked on a periodic
box with every dissipation off and the (P_i + P_j) force (the exact adjoint of
the continuity divergence), so the semi-discrete system conserves
E = 1/2 sum m|v|^2 + sum m c^2 (rho - rho0)^2 / 2.
"""
import numpy as np
import pytest
import torch


def _acousticEnergyRatio(timeCentred: bool, nx=32, courant=0.6, lamDx=6.0, nSteps=150, eps=1e-4,
                         integrator='symplecticEuler'):
    from warpSPH.runner import run
    from warpSPH.cases.tgvWeaklyCompressible import tgvWeaklyCompressibleCase as case
    from warpSPH.enumTypes import PressureForceScheme

    c0 = 10.0
    dx = 1.0 / nx
    k = 2 * np.pi * max(1, round(1.0 / (lamDx * dx)))
    energies = []
    cfg0, ic0, diag0 = case.configureScheme, case.initialConditions, case.diagnostics

    def cfg(ctx):
        cfg0(ctx)
        sc = ctx.schemeConfig
        sc.diffusionParams.inviscidAlpha = 0.0
        sc.diffusionParams.densityDelta = 0.0
        sc.shiftProperties.active = False
        sc.pressureForceTerm = PressureForceScheme.nonConservative
        sc.timeCentredContinuity = timeCentred

    def ic(ctx, system):
        ic0(ctx, system)
        system.state.velocities.zero_()
        system.state.densities = 1.0 + eps * torch.cos(k * system.state.positions[:, 0])

    def diag(ctx, system):
        st = system.state
        fl = st.kinds == 0
        m, v, r = st.masses[fl], st.velocities[fl], st.densities[fl]
        energies.append(float(0.5 * (m * (v * v).sum(-1)).sum() + (m * c0 ** 2 * (r - 1.0) ** 2 / 2).sum()))
        return {}

    case.configureScheme, case.initialConditions, case.diagnostics = cfg, ic, diag
    try:
        run(case, scheme='deltaSPH', integrationScheme=integrator, nx=nx, L=1.0,
            nSteps=nSteps, quiet=True, progress=False, plot=False, video=False,
            cudaGraph=False, pipelineOutputs=False, adaptiveDt=False,
            params=dict(soundSpeed=c0, targetDt=courant * dx / c0, shuffleIters=0, jitter=0.0,
                        inviscid=True, alpha=0.0, nu=0.0, k=1, uMag=0.0, shifting=False))
    finally:
        case.configureScheme, case.initialConditions, case.diagnostics = cfg0, ic0, diag0
    return energies[-1] / energies[1]


def test_symplecticEulerPumpsAcousticEnergyWithoutTheCorrection(runtime):
    """The explicit-midpoint (rho, v) update: measured 5.8e-3 per step at
    omega dt = 0.40 (h^4/4 = 6.4e-3), so > 2x over 150 steps."""
    assert _acousticEnergyRatio(False) > 1.5


@pytest.mark.parametrize('integrator', ['symplecticEuler', 'velocityVerlet', 'leapFrog', 'pefrl', 'vefrl'])
def test_timeCentredContinuityConservesAcousticEnergy(runtime, integrator):
    """Density as a drift field (warpSPHIntegrators/drift.py): every second-order
    Verlet-family integrator keeps the acoustic energy (without it they grow at
    ~h^2 per step, symplecticEuler at h^4/4)."""
    ratio = _acousticEnergyRatio(True, integrator=integrator)
    assert abs(np.log(ratio)) < 0.05
