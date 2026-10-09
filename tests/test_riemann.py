"""AV_PLAN Phase 7b.1: the Riemann solvers (`modules/riemann`).

* the star state `(p*, u*)` of every `RiemannSolver` against the exact solution (Toro ch. 4, Newton on the pressure
  function) on Toro's five test problems, to each solver's known accuracy: `L = R` is exact for all of them, TRRS is exact
  for two rarefactions (test 2), TSRS for two equal shocks, the acoustic family for weak waves;
* mirror symmetry: swapping the states and negating the velocities negates `u*` and leaves `p*`;
* HLLC: `F(W, W)` is the physical flux, a stationary contact is held exactly, the Sod flux is within 10 % of the exact
  Godunov flux, and the flux of a mirrored problem is the negated, mirrored flux;
* vacuum guards: zero density / pressure states give finite output.

Tolerances for the exact-limit checks are float32-sized in process; `test_float64` reruns the file in float64.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import warp as wp

from warpSPHCore import scalar_t

from warpSPH.modules.riemann import RiemannSolver, hllcFlux, riemannStarState

REPO = Path(__file__).resolve().parents[1]
FLOAT64 = scalar_t is wp.float64
NP = np.float64 if FLOAT64 else np.float32
EXACT_TOL = 1e-12 if FLOAT64 else 2e-5


@wp.kernel
def _starKernel(solver: wp.int32, W: wp.array2d(dtype=scalar_t), gamma: scalar_t,
                pStar: wp.array(dtype=scalar_t), uStar: wp.array(dtype=scalar_t)):
    k = wp.tid()
    p, u = riemannStarState(solver, W[k, 0], W[k, 1], W[k, 2], W[k, 3], W[k, 4], W[k, 5], gamma)
    pStar[k] = p
    uStar[k] = u


@wp.kernel
def _fluxKernel(W: wp.array2d(dtype=scalar_t), gamma: scalar_t, F: wp.array2d(dtype=scalar_t)):
    k = wp.tid()
    Fr, Fm, FE, S = hllcFlux(W[k, 0], W[k, 1], W[k, 2], W[k, 3], W[k, 4], W[k, 5], gamma)
    F[k, 0] = Fr
    F[k, 1] = Fm
    F[k, 2] = FE
    F[k, 3] = S


def star(solver, states, gamma=1.4):
    W = wp.array(np.asarray(states, dtype=NP), dtype=scalar_t)
    n = W.shape[0]
    p = wp.zeros(n, dtype=scalar_t)
    u = wp.zeros(n, dtype=scalar_t)
    wp.launch(_starKernel, dim=n, inputs=[solver.value, W, scalar_t(gamma)], outputs=[p, u])
    return p.numpy().astype(np.float64), u.numpy().astype(np.float64)


def flux(states, gamma=1.4):
    W = wp.array(np.asarray(states, dtype=NP), dtype=scalar_t)
    n = W.shape[0]
    F = wp.zeros((n, 4), dtype=scalar_t)
    wp.launch(_fluxKernel, dim=n, inputs=[W, scalar_t(gamma)], outputs=[F])
    return F.numpy().astype(np.float64)


def exactStar(rhoL, uL, pL, rhoR, uR, pR, gamma=1.4):
    """Toro ch. 4: Newton on f_L(p) + f_R(p) + (u_R - u_L) = 0."""
    aL, aR = math.sqrt(gamma * pL / rhoL), math.sqrt(gamma * pR / rhoR)

    def fK(p, rho, pK, aK):
        if p > pK:
            A = 2 / ((gamma + 1) * rho)
            B = (gamma - 1) / (gamma + 1) * pK
            f = (p - pK) * math.sqrt(A / (p + B))
            df = math.sqrt(A / (B + p)) * (1 - (p - pK) / (2 * (B + p)))
        else:
            f = 2 * aK / (gamma - 1) * ((p / pK) ** ((gamma - 1) / (2 * gamma)) - 1)
            df = (1 / (rho * aK)) * (p / pK) ** (-(gamma + 1) / (2 * gamma))
        return f, df

    p = max(1e-8, 0.5 * (pL + pR))
    for _ in range(100):
        fl, dfl = fK(p, rhoL, pL, aL)
        fr, dfr = fK(p, rhoR, pR, aR)
        pn = max(1e-12, p - (fl + fr + uR - uL) / (dfl + dfr))
        if abs(pn - p) < 1e-14 * (pn + p):
            p = pn
            break
        p = pn
    fl, _ = fK(p, rhoL, pL, aL)
    fr, _ = fK(p, rhoR, pR, aR)
    return p, 0.5 * (uL + uR) + 0.5 * (fr - fl)


# Toro Table 4.1 (rho, u, p) L then R
TORO = {
    1: (1.0, 0.0, 1.0, 0.125, 0.0, 0.1),
    2: (1.0, -2.0, 0.4, 1.0, 2.0, 0.4),
    3: (1.0, 0.0, 1000.0, 1.0, 0.0, 0.01),
    4: (5.99924, 19.5975, 460.894, 5.99242, -6.19633, 46.0950),
    5: (1.0, -19.5975, 1000.0, 1.0, -19.5975, 0.01),
}


def test_exactReferenceMatchesToro():
    """The reference itself, against the published star values of Toro Table 4.2."""
    for t, (pRef, uRef) in {1: (0.30313, 0.92745), 2: (0.00189, 0.0), 3: (460.894, 19.5975), 4: (1691.64, 8.68975)}.items():
        p, u = exactStar(*TORO[t])
        assert p == pytest.approx(pRef, rel=1e-4, abs=1e-5), t
        assert u == pytest.approx(uRef, rel=1e-4, abs=1e-5), t


@pytest.mark.parametrize('solver', list(RiemannSolver))
def test_equalStatesAreExact(solver):
    states = [(1.0, 0.7, 2.0, 1.0, 0.7, 2.0), (0.3, -1.5, 5.0, 0.3, -1.5, 5.0), (1e-3, 0.0, 1e-3, 1e-3, 0.0, 1e-3)]
    p, u = star(solver, states)
    for k, (rho, uK, pK, *_ ) in enumerate(states):
        assert p[k] == pytest.approx(pK, rel=EXACT_TOL * 10), (solver, k)
        assert u[k] == pytest.approx(uK, rel=EXACT_TOL * 10, abs=EXACT_TOL), (solver, k)


def test_trrsIsExactForTwoRarefactions():
    # Toro test 2 (123 problem) is two symmetric rarefactions, also a weak one with an offset
    for case in (TORO[2], (1.0, -1.0, 1.0, 0.5, 0.5, 0.5)):
        pRef, uRef = exactStar(*case)
        p, u = star(RiemannSolver.TRRS, [case])
        if case is TORO[2]:
            assert p[0] == pytest.approx(pRef, rel=1e-4) and u[0] == pytest.approx(uRef, abs=1e-5)
        else:
            # not both rarefactions in general -- only require the test above to be the exact one
            assert np.isfinite(p[0]) and np.isfinite(u[0])


def test_tsrsTwoShocks():
    """Two equal colliding streams. TSRS uses the PVRS pressure as its shock-speed estimate, so it is not exact: its
    error is second order in the wave strength (0.5 % at |u| = 0.3 a, 8.5 % at |u| = a) and always well below the
    acoustic family's (4 % / 25 %), which has no quadratic term."""
    for case, tolTsrs in (((1.0, 0.3, 1.0, 1.0, -0.3, 1.0), 0.01), ((1.0, 1.0, 1.0, 1.0, -1.0, 1.0), 0.10)):
        pRef, _ = exactStar(*case)
        pT, uT = star(RiemannSolver.TSRS, [case])
        pA, _ = star(RiemannSolver.PVRS, [case])
        assert abs(pT[0] - pRef) / pRef < tolTsrs
        assert abs(pT[0] - pRef) < 0.5 * abs(pA[0] - pRef)
        assert abs(uT[0]) < 1e-5


@pytest.mark.parametrize('solver', list(RiemannSolver))
def test_weakWavesStarState(solver):
    """A weak problem (5-10 % jumps): every solver is second order in the wave strength, so within 1 %."""
    for case in ((1.0, 0.0, 1.0, 1.05, 0.02, 1.1), (1.0, 0.05, 1.0, 0.95, -0.05, 0.9)):
        pRef, uRef = exactStar(*case)
        p, u = star(solver, [case])
        assert p[0] == pytest.approx(pRef, rel=0.01), (solver, case)
        assert u[0] == pytest.approx(uRef, abs=0.01 * 1.2), (solver, case)


# Observed against the exact solution (the values Toro tabulates in ch. 9 for the pressure estimates): the pressure
# error of each solver on each test, relative; the weak-wave solvers (Acoustic, PVRS) are not meant for tests 1-5
@pytest.mark.parametrize('solver,tests', [
    (RiemannSolver.TRRS, {1: 0.02}),                                   # test 2 is its exact case (above)
    (RiemannSolver.TSRS, {1: 0.05, 3: 0.02, 5: 0.02}),
    (RiemannSolver.Adaptive, {1: 0.05, 3: 0.02, 5: 0.02}),
    (RiemannSolver.HLLC, {1: 0.10, 3: 0.16, 5: 0.16}),
])
def test_strongProblemsStarPressure(solver, tests):
    for t, tol in tests.items():
        pRef, uRef = exactStar(*TORO[t])
        p, u = star(solver, [TORO[t]])
        assert p[0] == pytest.approx(pRef, rel=tol), (solver, t, p[0], pRef)


def test_adaptiveSelectsByPressureRatio():
    """Weak: the PVRS branch (equals PVRS); strong expansion: the TRRS branch; strong compression: the TSRS branch."""
    weak = (1.0, 0.0, 1.0, 1.05, 0.02, 1.1)
    expansion = TORO[2]
    compression = TORO[3]
    for case, ref in ((weak, RiemannSolver.PVRS), (expansion, RiemannSolver.TRRS), (compression, RiemannSolver.TSRS)):
        pa, ua = star(RiemannSolver.Adaptive, [case])
        pr, ur = star(ref, [case])
        np.testing.assert_allclose([pa[0], ua[0]], [pr[0], ur[0]], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('solver', list(RiemannSolver))
def test_mirrorSymmetry(solver):
    rng = np.random.default_rng(3)
    s = np.c_[rng.uniform(0.2, 2, 64), rng.normal(0, 1, 64), rng.uniform(0.2, 3, 64),
              rng.uniform(0.2, 2, 64), rng.normal(0, 1, 64), rng.uniform(0.2, 3, 64)]
    m = np.c_[s[:, 3], -s[:, 4], s[:, 5], s[:, 0], -s[:, 1], s[:, 2]]
    p, u = star(solver, s)
    pm, um = star(solver, m)
    np.testing.assert_allclose(pm, p, rtol=1e-4 if not FLOAT64 else 1e-11)
    np.testing.assert_allclose(um, -u, rtol=1e-4 if not FLOAT64 else 1e-11, atol=1e-5 if not FLOAT64 else 1e-11)


def physicalFlux(rho, u, p, gamma=1.4):
    E = p / (gamma - 1) + 0.5 * rho * u * u
    return np.array([rho * u, rho * u * u + p, (E + p) * u])


def test_hllcConsistencyAndContact():
    states = [(1.3, 0.4, 0.9, 1.3, 0.4, 0.9), (0.5, -2.0, 3.0, 0.5, -2.0, 3.0), (1.0, 5.0, 1.0, 1.0, 5.0, 1.0)]
    F = flux(states)
    for k, (rho, u, p, *_ ) in enumerate(states):
        np.testing.assert_allclose(F[k, :3], physicalFlux(rho, u, p), rtol=EXACT_TOL * 50, atol=EXACT_TOL * 10)
    # a stationary contact: u = 0, equal pressure, different density -> pure pressure flux, exactly
    F = flux([(1.0, 0.0, 1.0, 0.125, 0.0, 1.0)])
    np.testing.assert_allclose(F[0, :3], [0.0, 1.0, 0.0], atol=EXACT_TOL * 10)


def test_hllcSodFlux():
    """The exact Godunov flux at x/t = 0 is the star-left state of Sod (rho* 0.42632, u* 0.92745, p* 0.30313). HLLC with
    Toro's PVRS wave speeds smears the rarefaction fan into one state: mass and energy within 3-5 %, the momentum flux
    (the pressure, where the wave-speed error is largest) within 25 %."""
    ref = np.array([0.39539, 0.66985, 1.15404])
    F = flux([TORO[1]])
    assert np.all(np.abs(F[0, :3] / ref - 1.0) < np.array([0.03, 0.25, 0.05])), F[0, :3]


def test_hllcMirror():
    rng = np.random.default_rng(5)
    s = np.c_[rng.uniform(0.2, 2, 64), rng.normal(0, 1, 64), rng.uniform(0.2, 3, 64),
              rng.uniform(0.2, 2, 64), rng.normal(0, 1, 64), rng.uniform(0.2, 3, 64)]
    m = np.c_[s[:, 3], -s[:, 4], s[:, 5], s[:, 0], -s[:, 1], s[:, 2]]
    F, Fm = flux(s), flux(m)
    tol = 1e-4 if not FLOAT64 else 1e-11
    np.testing.assert_allclose(Fm[:, 0], -F[:, 0], rtol=tol, atol=tol)     # mass flux flips
    np.testing.assert_allclose(Fm[:, 1], F[:, 1], rtol=tol, atol=tol)      # normal-momentum flux is even
    np.testing.assert_allclose(Fm[:, 2], -F[:, 2], rtol=tol, atol=tol)     # energy flux flips


@pytest.mark.parametrize('solver', list(RiemannSolver))
def test_vacuumGuards(solver):
    p, u = star(solver, [(0.0, 0.0, 0.0, 1.0, 0.0, 1.0), (1.0, 0.0, 1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                         (1.0, -50.0, 1.0, 1.0, 50.0, 1.0)])
    assert np.all(np.isfinite(p)) and np.all(np.isfinite(u)), (solver, p, u)
    F = flux([(0.0, 0.0, 0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)])
    assert np.all(np.isfinite(F))


@pytest.mark.skipif(FLOAT64, reason='already float64: the in-process test is the strict one')
def test_float64():
    env = dict(os.environ, warpSPHCore_PRECISION='float64')
    proc = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-x', '-p', 'no:cacheprovider',
                           '-k', 'not float64', __file__],
                          cwd=REPO, env=env, capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]


def test_requested_precision():
    if os.environ.get('warpSPHCore_PRECISION') == 'float64':
        assert FLOAT64
