"""The switch registry (`modules/shockCapturing/wrapper.py`): every `ViscositySwitch` member
either resolves to an implementation or raises a NAMED `NotImplementedError`."""

from __future__ import annotations

import pytest

from warpSPH.enumTypes import ViscositySwitch
from warpSPH.modules.shockCapturing.wrapper import (
    PLANNED, SWITCHES, _resolve, registerViscositySwitch)


@pytest.mark.parametrize('scheme', list(ViscositySwitch))
def test_every_member_resolves_or_raises_a_named_not_implemented(scheme):
    if scheme in SWITCHES:
        termsFn, updateFn = _resolve(scheme)
        assert callable(termsFn) and callable(updateFn)
    else:
        assert scheme in PLANNED, f'{scheme} is neither implemented nor planned'
        with pytest.raises(NotImplementedError, match='AV_PLAN'):
            _resolve(scheme)


def test_registered_and_planned_are_disjoint_and_cover_the_enum():
    assert not set(SWITCHES) & set(PLANNED)
    assert set(SWITCHES) | set(PLANNED) == set(ViscositySwitch)


def test_register_moves_a_planned_member_into_the_registry():
    scheme = ViscositySwitch.Colagrossi2004
    saved = SWITCHES.pop(scheme)
    PLANNED[scheme] = 'AV_PLAN test placeholder'
    try:
        registerViscositySwitch(scheme, *saved)
        assert scheme in SWITCHES and scheme not in PLANNED
    finally:
        SWITCHES[scheme] = saved
        PLANNED.pop(scheme, None)
