"""Geometric interface markers and volume-integrated, corrected SPH curl.

These are observation operators, independent of the force's CRK gradient.
Circulation is evaluated in a moving streamwise window about the marker band,
excluding the distant periodic seam. It is not a molecular mixing diagnostic.
"""
import torch


def markerMeasures(x, cos, sin):
    """Lagrangian marker-band envelope; Fourier amplitude is useful before folding.

    Equal numbers of initially adjacent rows on either side bracket the contact.
    Their finite initial separation contributes O(dx) to envelope widths, so
    report that offset and retain it in resolution comparisons.
    """
    mean = x.mean()
    return dict(interfaceMean=mean,
        interfaceCosAmplitude=2 * ((x - mean) * cos).mean(),
        interfaceSinAmplitude=2 * ((x - mean) * sin).mean(),
        interfaceMin=x.min(), interfaceMax=x.max(),
        # Light material 0 is on the left: its bubble advances in +x.
        bubbleExtent=x.max()-mean, spikeExtent=mean-x.min(),
        markerLayerWidth=x.max()-x.min())


def curlMeasures(state, jacobian, mask):
    """J_ab = dv_a/dx_b; physical particle volume is m/rho.

    Keeping both signed half-wave integrals avoids the zero total circulation
    of a symmetric single mode hiding two strong counter-rotating vortices.
    """
    omega = jacobian[:, 1, 0] - jacobian[:, 0, 1]
    volume = torch.where(mask, state.masses / state.densities, 0.)
    circulation = volume * omega
    upper = state.positions[:, 1] >= 0
    return dict(circulationUpper=torch.where(upper, circulation, 0.).sum(),
        circulationLower=torch.where(~upper, circulation, 0.).sum(),
        circulationPositive=circulation.clamp(min=0).sum(),
        circulationNegative=circulation.clamp(max=0).sum(),
        enstrophy=.5 * (volume * omega.square()).sum(),
        transverseKineticEnergy=.5 * (torch.where(mask, state.masses, 0.) * state.velocities[:, 1].square()).sum(),
        interactionAlphaMean=(volume * state.alphas).sum() / volume.sum(),
        interactionAlphaActiveVolume=(volume * (state.alphas > .1)).sum() / volume.sum()), omega


def interactionDiagnostics(ctx, system, markerX):
    """Expensive curl reductions at a physical-time cadence, with explicit age."""
    time = float(system.t)
    cached = ctx.scratch.get('rmInteractionDiagnostics')
    if cached is not None and time < ctx.scratch['rmNextDiagnosticTime']:
        return cached
    from ....modules.reconstruction.gradient import computeVelocityJacobian
    from warpSPHCore import SupportScheme
    state = system.state
    # Two support radii include the regularized shear layer near each marker.
    margin = 2 * state.supports.max()
    mask = ((state.positions[:, 0] >= markerX.min()-margin) &
            (state.positions[:, 0] <= markerX.max()+margin))
    jacobian = computeVelocityJacobian(state, ctx.config,
        SupportScheme.KernelMeanSymmetric, system.adjacency, corrected=True)
    values, _ = curlMeasures(state, jacobian, mask)
    values.update(circulationWindowMin=markerX.min()-margin,
                  circulationWindowMax=markerX.max()+margin)
    host = torch.stack([v.detach().reshape(()) for v in values.values()]).cpu().tolist()
    cached = dict(zip(values, host), circulationTime=time)
    ctx.scratch['rmInteractionDiagnostics'] = cached
    ctx.scratch['rmNextDiagnosticTime'] = time + ctx.param('rmDiagnosticInterval')
    return cached
