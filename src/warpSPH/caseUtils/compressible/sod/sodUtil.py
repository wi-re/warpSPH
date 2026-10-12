"""Six-panel Sod profile plotting (density, supports, velocity, thermal
energy, pressure, entropy vs. `x`), shared by `cases/compressible/sod.py` and the 2D/3D
`sodND` cases via the `scatter` flag -- since those share many particles at
the same `x`, a scatter reads better than a connecting line. `plotSod`
creates the figure, `plotSod_` redraws into an existing one for per-frame
updates; both optionally overlay `sodSolution.solve`'s exact Riemann
solution and label the five characteristic regions.
"""

__all__ = ['plotSod', 'plotSod_']


def decoratePlot(axis, positions, text = True):
    axis.axvline(positions['Head of Rarefaction'], color='black', linestyle='--', alpha = 0.5)
    axis.axvline(positions['Foot of Rarefaction'], color='black', linestyle='--', alpha = 0.5)
    axis.axvline(positions['Contact Discontinuity'], color='black', linestyle='--', alpha = 0.5)
    axis.axvline(positions['Shock'], color='black', linestyle='--', alpha = 0.5)
    if text:
        ymax = axis.get_ylim()[1]
        ymin = axis.get_ylim()[0]
        position = ymin + 0.85 * (ymax - ymin)
        axis.text(positions['Head of Rarefaction'] / 2, position, 'I')
        axis.text((positions['Head of Rarefaction'] + positions['Foot of Rarefaction']) / 2, position, 'II')
        axis.text((positions['Foot of Rarefaction'] + positions['Contact Discontinuity']) / 2, position, 'III')
        axis.text((positions['Contact Discontinuity'] + positions['Shock']) / 2, position, 'IV')
        axis.text((positions['Shock'] + 1) / 2, position, 'V')
        
        
from ....modules import idealGasEOS
from .sod import sodInitialState
import torch
import matplotlib.pyplot as plt
from .sodSolution import solve


def plotSod_(fig, axis, simulationState, simulationConfig, schemeConfig, domain, gamma, leftState: sodInitialState, rightState: sodInitialState, plotReference = True, plotLabels = True, scatter = False, t_ = None):
    referenceState = simulationState
    # neighborhood, sparseNeighborhood = buildNeighborhood(referenceState, referenceState, domain, verletScale = 1.0)
    # CSR = coo_to_csr(sparseNeighborhood)
    pos = referenceState.positions.cpu().numpy()
    densities = referenceState.densities.cpu().numpy()
    velocities = referenceState.velocities.cpu().numpy()
    thermalEnergy = referenceState.thermalEnergies.cpu().numpy() if hasattr(referenceState, 'thermalEnergies') else referenceState.internalEnergies.cpu().numpy()
    supports = referenceState.supports.cpu().numpy()
    pressures = referenceState.pressures.cpu().numpy() if hasattr(referenceState, 'pressures') and referenceState.pressures is not None else referenceState.P.cpu().numpy()
    # neighbors = CSR.rowEntries.cpu().numpy()
    masses = referenceState.masses.cpu().numpy()
    
    A_, u_, P_, c_ = idealGasEOS(A = None, u = torch.tensor(thermalEnergy, dtype = referenceState.densities.dtype, device = referenceState.densities.device), P = None, rho = referenceState.densities, gamma = gamma)
    A_ = A_.cpu().numpy()
    
    kineticEnergy_ = 0.5 * (torch.linalg.norm(simulationState.velocities, dim = -1) **2 * simulationState.masses).sum()
    thermalEnergy_ = (simulationState.internalEnergies * simulationState.masses).sum()
    totalEnergy = kineticEnergy_ + thermalEnergy_
        

    t = t_

    indices = torch.argsort(referenceState.positions[:,0]).cpu().numpy()
    indices = indices[pos[indices][:,0] > 0]

    simText = f'{schemeConfig.schemeName}, kernel: {simulationConfig.kernel.name}, support: {simulationConfig.supportMode.name}, integrator: {simulationConfig.integrationScheme.name}' + f', Energy: {schemeConfig.energyScheme.name}' if hasattr(schemeConfig, 'energyScheme') and schemeConfig.energyScheme is not None else ''

    fig.suptitle(f'{simText}\nnx = {pos.shape[0]//2}, t = {t:6.4g}, Kinetic = {kineticEnergy_.cpu().item():6.4g}, Thermal = {thermalEnergy_.cpu().item():6.4g}, Total = {totalEnergy.cpu().item():6.4g}')
    # Everything is plotted against x, and the shock runs along x, so 2D/3D
    # runs (`sodND`) drop straight into these panels as a scatter over the
    # transverse directions. Indexing the x column explicitly is what makes
    # that work; for dim=1 it is the same data these lines always plotted.
    x = pos[indices, 0]
    velocityX = velocities[indices, 0]
    if not scatter:
        axis[0,0].plot(x, densities[indices], label='Density')
        axis[0,1].plot(x, supports[indices], label='Supports')
        axis[0,2].plot(x, velocityX, label='Velocity')
        axis[1,0].plot(x, thermalEnergy[indices], label='Thermal Energy')
        axis[1,1].plot(x, pressures[indices], label='Pressure')
        axis[1,2].plot(x, A_[indices], label='Neighbors')
    else:
        s = 1
        axis[0,0].scatter(x, densities[indices], s = s, label='Density')
        axis[0,1].scatter(x, supports[indices], s = s, label='Supports')
        axis[0,2].scatter(x, velocityX, s = s, label='Velocity')
        axis[1,0].scatter(x, thermalEnergy[indices], s = s, label='Thermal Energy')
        axis[1,1].scatter(x, pressures[indices], s = s, label='Pressure')
        axis[1,2].scatter(x, A_[indices], s = s, label='Masses')
    
    axis[0,0].set_title('Density')
    axis[0,1].set_title('Supports')
    axis[0,2].set_title('Velocity')
    axis[1,0].set_title('Thermal Energy')
    axis[1,1].set_title('Pressure')
    axis[1,2].set_title('Entropy')
    if plotReference:
        dustFrac = 0.0
        npts = 500
        # t = 1e-4
        left_state = (leftState.p,leftState.rho,leftState.v)
        right_state = (rightState.p, rightState.rho, rightState.v)
        # right_state = (0.1, 0.5, 0.)

        positions, regions, values = solve(left_state=left_state, \
            right_state=right_state, geometry=(0., 1., 0.5), t=t.cpu().item() if isinstance(t, torch.Tensor) else t,
            gamma=gamma, npts=npts, dustFrac=dustFrac)
        axis[0,0].plot(values['x'], values['rho'], label='Reference Solution', alpha = 0.5, ls = ':', c = 'black')
        axis[1,1].plot(values['x'], values['p'], label='Reference Solution', alpha = 0.5, ls = ':', c = 'black')
        axis[0,2].plot(values['x'], values['u'], label='Reference Solution', alpha = 0.5, ls = ':', c = 'black')
        axis[1,0].plot(values['x'], values['energy'], label='Reference Solution', alpha = 0.5, ls = ':', c = 'black')
            
        if plotLabels:
            decoratePlot(axis[0,0], positions, text = True)
            decoratePlot(axis[0,1], positions, text = False)
            decoratePlot(axis[0,2], positions, text = True)
            decoratePlot(axis[1,0], positions, text = False)
            decoratePlot(axis[1,1], positions, text = True)
            decoratePlot(axis[1,2], positions, text = False)


def plotSod(simulationState, simulationConfig, schemeConfig, domain, gamma, leftState: sodInitialState, rightState: sodInitialState, plotReference = True, plotLabels = True, scatter = False, t_ = None):
    fig, axis = plt.subplots(2, 3, figsize=(10, 5), squeeze=False, sharex=True, sharey=False)

    plotSod_(fig, axis, simulationState, simulationConfig, schemeConfig, domain, gamma, leftState, rightState, plotReference, plotLabels, scatter, t_)
    fig.tight_layout()

    return fig, axis