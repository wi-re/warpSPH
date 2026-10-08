"""GODUNOV_SPH_PLAN: the Cha, Inutsuka & Nayakshin (2010) Fig. 1 test, `scripts/probe_densityJumpForce.py`.

Particles at rest in exact pressure equilibrium across a 4:1 density jump: the exact acceleration is zero. The standard SPH momentum equation gives an
O(1) repulsion (in units of `P0 / (rho h)`) at the jump; the simplified Godunov SPH (Cha & Whitworth) is identically that force at constant pressure
and shares it. Inutsuka's convolution form (layer 3) is the scheme whose force must vanish: its assertion is skipped until it exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import warp as wp

from warpSPHCore import scalar_t

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
import probe_densityJumpForce as P   # noqa: E402

FLOAT64 = scalar_t is wp.float64


@pytest.fixture(scope='module')
def state():
    return P.contactState(4.0)


@pytest.fixture(scope='module')
def stateConstH():
    """2:1 jump with a spatially constant smoothing length: the case Inutsuka's derivation is exact for (a varying h adds the h_i / h_j halves
    approximation on top of the volume interpolation)."""
    return P.contactState(2.0, uniformSupport=True)


def test_standardSPHHasASpuriousForceAtTheJump(state):
    system, config = state
    jump, bulk = P.spuriousForce(P.sphForce(system, config), system)
    assert jump > 0.5, 'the standard SPH force should not vanish at a density jump (Cha et al. 2010)'
    assert bulk < 1e-2


@pytest.mark.parametrize('order', [1, 2])
def test_simplifiedGSPHSharesIt(state, order):
    system, config = state
    sph_jump, _ = P.spuriousForce(P.sphForce(system, config), system)
    jump, bulk = P.spuriousForce(P.gsphForce(system, config, order), system)
    assert jump == pytest.approx(sph_jump, rel=0.3), 'the simplified form is the standard SPH force at constant pressure'
    assert bulk < 1e-2


@pytest.mark.parametrize('cubic,factor', [(True, 0.35), (False, 0.5)])
def test_inutsukaConvolutionFormReducesTheForce_wideGaussian(stateConstH, cubic, factor):
    """With a Gaussian wider than the spacing (h = support / 3, ~1.3 spacings) Inutsuka's force at a 2:1 jump (constant h) is well below the standard SPH
    one: 0.22x with the cubic volume interpolant, 0.34x with the linear one, and stays at lattice-noise level in the bulk. It is not zero: with the exact
    `1/rho^2` it would be (the identity of Inutsuka 2002 Eq. 23, checked by quadrature, `scripts/probe_densityJumpQuadrature.py`); the interpolant along the
    pair axis across a sharp jump is the limit."""
    system, config = stateConstH
    sph_jump, _ = P.spuriousForce(P.sphForce(system, config), system)
    a, rho = P.inutsukaForce(system, config, cubic=cubic, hMode='support')
    st = system.state
    saved = st.densities
    st.densities = rho                   # the scale of the measure is the Gaussian density's
    jump, bulk = P.spuriousForce(a, system)
    st.densities = saved
    assert jump < factor * sph_jump, (jump, sph_jump)
    assert bulk < 1e-2


def test_inutsukaWithTheSchemesGaussianIsAModestImprovement(stateConstH):
    """With the scheme's Gaussian (h = the local spacing, the pairing-stable choice) the consistency gain is much smaller: ~0.78x at 2:1."""
    system, config = stateConstH
    sph_jump, _ = P.spuriousForce(P.sphForce(system, config), system)
    a, rho = P.inutsukaForce(system, config, cubic=True)
    st = system.state
    saved = st.densities
    st.densities = rho
    jump, bulk = P.spuriousForce(a, system)
    st.densities = saved
    assert jump < 0.95 * sph_jump and bulk < 1e-2


def test_inutsukaBeatsSPHWithAdaptiveSupportToo(state):
    system, config = state
    sph_jump, _ = P.spuriousForce(P.sphForce(system, config), system)
    a, rho = P.inutsukaForce(system, config, cubic=True, hMode='support')
    st = system.state
    saved = st.densities
    st.densities = rho
    jump, bulk = P.spuriousForce(a, system)
    st.densities = saved
    assert jump < 0.9 * sph_jump
