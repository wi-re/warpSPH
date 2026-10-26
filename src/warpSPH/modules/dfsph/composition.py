"""Opt-in IISPH-based predictor-first and shift/resample compositions.

The standard DF-first route remains in the scheme module. Physical and
auxiliary shift pressures are exposed separately on each returned update.
"""
from types import SimpleNamespace as NS
import torch
from warpSPHCore import (SupportScheme,buildVerletList,OperationProperties,OperationDirection,
    WarpOperation,GradientScheme,warpOperation)
from warpSPH.configurations.compute import neighborSearchOptions
from warpSPH.modules.boundaryConditions import enforceDirichlet,computeForcing,enforceUpdates
from warpSPH.modules.density import computeDensities
from warpSPH.modules.gravity import computeGravity,computeBodyForce
from warpSPH.modules.deltaSPH import computeVelocityDiffusion
from warpSPH.modules.iisph import (computeIISPHContactPoints,computeIISPHProjectionGeometry,
    computeIISPHBoundaryPressureAcceleration,assembleIISPHUpdate,maskIISPHUpdate)
from warpSPH.modules.iisph.solver import IISPHResult
from warpSPH.modules.dfsph import solveDFSPHProjection
from warpSPH.modules.incompressible.stages import requireProjectedStep
from warpSPH.modules.util import countNeighbors
from warpSPHIntegrators.drivers.mechanics.projection import kick,drift,symplecticProjection


def geometry(system,cfg,sc,*,dt=None):
    st=system.state
    adj=buildVerletList(st,cfg.domain,verletScale=cfg.verletScale,supportMode=SupportScheme.SuperSymmetric,
        priorNeighborhood=system.adjacency,**neighborSearchOptions(cfg))
    system.adjacency=adj
    st.densities=computeDensities(st,cfg,sc,adj,boundaryRepresentation=sc.iisph.boundaryRepresentation,
        wallMassDensity=sc.fluid.restDensity,wallSupport=system.boundarySupport)
    # Physical-stage coefficients and the sampling gate must see prescribed
    # density, as in IISPH and the standard DF-first step. The post-drift
    # auxiliary shift instead measures density on its new geometry.
    if dt is not None:
        enforceDirichlet(system,system.t,dt,cfg,sc)
    cp,coverage=computeIISPHContactPoints(st,system.contactBoundaryProvider)
    out=computeIISPHProjectionGeometry(st,cfg,sc,adj,analyticProvider=system.analyticBoundaryProvider,
        boundarySupport=system.boundarySupport,contactPoints=cp,contactCoverageFailure=coverage)
    return NS(adjacency=adj,args=out[:5],integrals=out[7],contactPoints=cp,
        contactAdjacency=out[5],contactDenominator=out[6])


def solve(st,cfg,sc,dt,g,mode,velocity,history):
    deficient=None
    if mode=='divergence' and sc.dfsph.deficiencyGuard and sc.dfsph.divergence.clampPressure:
        deficient=countNeighbors(st,cfg,sc,g.adjacency)<(7 if cfg.domain.dim==2 else 20)
    return solveDFSPHProjection(st,cfg,sc,dt,g.adjacency,*g.args,mode=mode,velocity=velocity,
        pressureImpulse=history,contactPoints=g.contactPoints,contactAdjacency=g.contactAdjacency,
        contactDenominator=g.contactDenominator,deficient=deficient)


def advanceDFSPH(system,dt,cfg,sc,order,*,fluidOnlyViscosity=False):
    if order not in ('df_np_cd', 'np_df_cd', 'np_df_shift'):
        raise ValueError(f'Unknown DFSPH composition: {order}')
    requireProjectedStep(system,cfg,'dfsph')
    st=system.state;fluid=st.kinds==0;mask=fluid[:,None]
    g=geometry(system,cfg,sc,dt=dt)
    events=[]

    def project_div():
        events.append('divergence')
        before=st.velocities
        result=solve(st,cfg,sc,dt,g,'divergence',before,st.divergencePressureImpulse)
        after=kick(before,dt,result.acceleration,mask=mask)
        st.velocities=after
        return result

    if order=='df_np_cd':divergence=project_div()
    events.append('nonpressure')
    gravity=computeGravity(st,cfg,sc,g.adjacency)
    kwargs=dict(boundaryRepresentation=sc.iisph.boundaryRepresentation,wallSupport=system.boundarySupport,
        boundaryVelocitiesPrepared=False)
    if fluidOnlyViscosity:kwargs.update(operationMode=OperationDirection.FluidToFluid,wall=False)
    nonpressure=gravity+computeBodyForce(st,sc)+computeVelocityDiffusion(st,cfg,sc,g.adjacency,**kwargs)
    nonpressure=nonpressure+computeForcing(system,dt,system.t,cfg,sc)/st.masses[:,None]
    known=computeIISPHBoundaryPressureAcceleration(st,cfg,g.adjacency,g.args[0],g.args[2],g.integrals,
        g.contactPoints,gravity,contactGravity=sc.iisph.contactGravity,massFactor=float(getattr(sc,'analyticWallMass',1.)))
    # Same affine wall-gravity force, once per physical step. In predictor-first
    # compositions it belongs in the predictor so DF can balance it physically.
    physical=torch.where(mask,nonpressure+known,0)
    if order!='df_np_cd':
        before=st.velocities
        st.velocities=kick(before,dt,physical,mask=mask)
        divergence=project_div()

    zero=IISPHResult(torch.zeros_like(st.masses),torch.zeros_like(st.velocities),0,0.,'disabled',torch.zeros_like(st.masses))
    density=zero;shift=torch.zeros_like(st.velocities);resample=torch.zeros_like(st.velocities)
    if order in ('df_np_cd','np_df_cd'):
        events.append('density_velocity')
        predictor=st.velocities+dt*physical if order=='df_np_cd' else st.velocities
        density=solve(st,cfg,sc,dt,g,'density',predictor,st.densityPressureImpulse)
        total=physical+density.acceleration if order=='df_np_cd' else density.acceleration
        update=assembleIISPHUpdate(st,total,torch.zeros_like(total))
        enforceUpdates(update,system,dt,system.t,cfg,sc);maskIISPHUpdate(update,st.kinds)
        symplecticProjection(st,dt,update.dvdt,mask=mask)
        events.append('drift')
    else:
        update=assembleIISPHUpdate(st,torch.zeros_like(st.velocities),torch.zeros_like(st.velocities))
        enforceUpdates(update,system,dt,system.t,cfg,sc);maskIISPHUpdate(update,st.kinds)
        st.velocities=kick(st.velocities,dt,update.dvdt,mask=mask)
        st.positions=drift(st.positions,dt,st.velocities,mask=mask)
        events.append('drift')
        st.positions = wrapPositions(st.positions, cfg.domain)
        if order=='np_df_shift':
            events.append('rebuild_after_drift')
            gshift=geometry(system,cfg,sc)
            # Already drifted: source is 1-rho(x_advected)/rho0, with no second
            # velocity prediction and no physical wall gravity in the shift.
            density=solve(st,cfg,sc,dt,gshift,'density',torch.zeros_like(st.velocities),st.densityPressureImpulse)
            shift=torch.where(mask,dt*dt*density.acceleration,0)
            gradient=warpOperation(st,OperationProperties(operation=WarpOperation.Gradient,kernel=cfg.kernel,
                supportMode=SupportScheme.Gather,operationMode=OperationDirection.FluidToFluid,
                gradientMode=GradientScheme.Difference),queryValues=st.velocities,domain=cfg.domain,adjacency=gshift.adjacency)
            resample=torch.einsum('nij,nj->ni',gradient,shift)
            st.positions=st.positions+shift
            st.velocities=st.velocities+torch.where(mask,resample,0)
            events.extend(['density_position','velocity_resample'])
    st.positions = wrapPositions(st.positions, cfg.domain)
    st.pressures=density.pressure;st.divergencePressures=divergence.pressure
    st.densityPressureImpulse=density.pressure*(dt*dt);st.divergencePressureImpulse=divergence.pressure*dt
    update.divergenceResult=divergence;update.densityResult=density
    update.orderAudit={'events':events,'order':order}
    return update,system.adjacency,st,([],[density.error,divergence.error])


def wrapPositions(positions, domain):
    return torch.where(domain.periodic, torch.remainder(positions-domain.min, domain.max-domain.min)+domain.min, positions)
