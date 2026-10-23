"""Known IISPH wall acceleration from hydrostatic pressure extrapolation."""

import torch
from .operators import launchIISPHKernel, pairArguments, geometryParameters


def computeIISPHBoundaryPressureAcceleration(state, config, schemeConfig, coefficients,
                                            boundaryIntegrals, contactPoints, gravity):
    """Integrate the gravity extension of q_solid = q_contact + g·(x_solid-contact).

    q is pressure divided by rest density. The contact-pressure unknown is
    handled by the linear pressure operator, independently of this known
    acceleration. Analytic integrals remain raw until used in this equation.
    """
    acceleration = torch.zeros_like(state.positions)
    if contactPoints is None or gravity is None or not schemeConfig.iisph.contactGravity:
        return acceleration
    if boundaryIntegrals is None:
        launchIISPHKernel('sampled_gravity', state, pairArguments(coefficients.pairs) +
            [coefficients.restVolumes, state.kinds.to(torch.int32).contiguous(),
             state.positions, contactPoints.contiguous(), gravity,
             *geometryParameters(state, config)[:2], acceleration])
        return acceleration

    # The provider calls this Cov: M_ab = integral_solid
    # (x_solid - x_i)_a * (grad_i W)_b dV, a first gradient moment.
    # It is not a statistical covariance. Summation combines the solid bodies;
    # analyticWallMass converts their geometric volume to calibrated rest volume.
    massFactor = float(getattr(schemeConfig, 'analyticWallMass', 1.0))
    firstGradientMoment = massFactor * boundaryIntegrals.out['Cov'].sum(0).to(state.positions)
    acceleration = (-torch.einsum('na,nab->nb', gravity, firstGradientMoment)
        - ((state.positions-contactPoints)*gravity).sum(-1, keepdim=True)*coefficients.wallGradient)
    return torch.where((state.kinds == 0)[:, None], acceleration, 0)
