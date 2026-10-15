"""The exact-delta integrator contract, in one place.

Some steps own their whole time step (the MFM / MFV meshless steps; the ACSPH dual-time step) and hand the integrator the *exact* per-step
delta ``(x^{n+1} - x^n) / dt``. Forward Euler applied to an exact delta is the identity, so the runner reproduces the converged state byte for
byte and no integrator needs to know about the scheme. That only holds for a **single-evaluation** integrator: a multi-stage one (RK2, RK4, ...)
would run the step once per stage and *blend* the results -- a plausible-looking but wrong update. So such a step

* has the runner pin the integrator up front (`pinExactDeltaIntegrator`, so the banner reports what actually runs), and
* refuses anything else itself at step entry (`requireExactDeltaIntegrator`).

`forwardEuler` and `explicitEuler` are the two integrators that evaluate the right-hand side once and apply it with a unit weight.
(ACSPH keeps its own copy of this tuple in `configurations/artificialCompressible.py` for now.)
"""

from warpSPHIntegrators.schemes import IntegrationSchemeType

__all__ = ['EXACT_DELTA_INTEGRATORS', 'EXACT_DELTA_INTEGRATOR_NAMES', 'pinExactDeltaIntegrator', 'requireExactDeltaIntegrator']

#: the integrators that evaluate the step exactly once per time step
EXACT_DELTA_INTEGRATORS = frozenset({IntegrationSchemeType.forwardEuler, IntegrationSchemeType.explicitEuler})
EXACT_DELTA_INTEGRATOR_NAMES = tuple(sorted(i.name for i in EXACT_DELTA_INTEGRATORS))


def pinExactDeltaIntegrator(requested: str) -> str:
    """The integrator name a run of an exact-delta scheme uses: `requested` if it is single-evaluation, else `forwardEuler`."""
    return requested if requested in EXACT_DELTA_INTEGRATOR_NAMES else 'forwardEuler'


def requireExactDeltaIntegrator(config, schemeName: str) -> None:
    """Refuse any integrator that would run the step more than once per time step."""
    scheme = getattr(config, 'integrationScheme', None)
    if scheme in EXACT_DELTA_INTEGRATORS:
        return
    raise ValueError(
        f'the {schemeName} step owns its full time step and must run under a single-evaluation integrator '
        f'({" / ".join(EXACT_DELTA_INTEGRATOR_NAMES)}); got {scheme!r}. Set config.integrationScheme to '
        f'IntegrationSchemeType.forwardEuler.')
