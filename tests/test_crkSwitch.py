"""The viscosity switch works under CRKSPH (AV_PLAN Phase 1 clean-up, 2026-10-07).

Two defects hid each other: CRK's `Q_i` / `Q_j` ignored the switched alpha altogether, and the scheme never wrote the switch's
`alpha0s` back, so under C&D alpha sat at ~0.97-0.98 (the initial 1, decayed by one step each step). Either alone made
`crkCullenDehnen2010` indistinguishable from `crkNone`."""

from __future__ import annotations

import numpy as np

from warpSPH.runner import run


def _sod(switch: str, nSteps: int = 150):
    from warpSPH.cases.sod import sodCase
    return run(sodCase, scheme='CRKSPH', nx=100, nSteps=nSteps, progress=False, quiet=True,
               params=dict(viscositySwitch=switch))


def test_cullen_dehnen_alpha_relaxes_away_from_its_initial_value_under_crk():
    alphas = _sod('CullenDehnen2010').state.state.alphas.detach().cpu().numpy()
    # frozen (the bug): ~0.97 everywhere. Working: most of the box has relaxed well below the initial 1 (median 0.13 here),
    # while the shock / contact region stays up (90th percentile 0.48)
    assert np.median(alphas) < 0.5
    assert np.percentile(alphas, 90) > 2.0 * np.median(alphas)


def test_none_switch_stays_at_one_under_crk():
    alphas = _sod('NoneSwitch').state.state.alphas.detach().cpu().numpy()
    assert np.all(alphas == 1.0)


def test_the_switch_changes_the_crk_viscosity():
    none = _sod('NoneSwitch').state.state
    cd = _sod('CullenDehnen2010').state.state
    # same initial state, same scheme: only the viscosity switch differs, so the states must differ measurably
    dv = (none.velocities - cd.velocities).abs().max().item()
    assert dv > 1e-5
