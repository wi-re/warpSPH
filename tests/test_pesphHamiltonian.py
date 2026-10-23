"""Real-SPH Hamiltonian qualification, independent dense potential and canonical probes.

Small derivative/reversal loops intentionally have no movies. Full smooth-flow
trajectories are recorded separately by diagnose_pesph_hamiltonian.py.
"""
from dataclasses import asdict, replace
import math
import time
import pytest
import torch
from test_pesphOperators import setup, cubic, DOUBLE
from warpSPH.enumTypes import AdaptiveSupportScheme, PESPHVariant
from warpSPH.modules.pesph import computePressureForce, computePressureState
from warpSPH.modules.pesph.stage import refreshPESPHState
from warpSPH.schemes.compressible.pesph import pesphConservativeStep
from warpSPH.systems.pesph import PESPHSystem
from warpSPH.utils.support import volumeToSupport
from warpSPHIntegrators import get_scheme
from warpSPHIntegrators.wrappers import classify_separable

pytestmark = pytest.mark.skipif(not DOUBLE, reason='Hamiltonian derivative and canonical-map checks require float64')


def fixture(dim=1, nx=24, adaptive=True):
    state, cfg, sc = setup(dim=dim, nx=nx)
    if adaptive:
        sc = replace(sc, adaptiveSupportScheme=AdaptiveSupportScheme.NumberDensity,
                     adaptiveSupportCorrections=True, adaptiveSupportIterations=64,
                     adaptiveSupportThreshold=1e-12)
    return PESPHSystem(state=state, domain=cfg.domain), cfg, sc


def clone(system):
    # A fresh list deliberately exercises the solve/cache lifecycle independently.
    return PESPHSystem(state=system.state.clone(), domain=system.domain, t=system.t)


def close(system, cfg, sc):
    p = refreshPESPHState(system, cfg, sc)
    acc, _ = computePressureForce(system.state, p, cfg, system.adjacency)
    return p, system.state.masses[:, None]*acc


def potential(state, cfg, sc):
    W, _ = cubic(state, cfg)
    w = state.masses*state.entropies.pow(1/sc.gamma)
    return (w*(W@w).pow(sc.gamma-1)).sum()/(sc.gamma-1)


def independent_gradient(system, cfg, sc):
    """Partial derivatives plus implicit dh/dq; no production force/grad-h used.

    Each C_i=n_i h_i^d-eta^d depends on its own h_i. Thus
    dU/dq = U_q - sum_i (U_hi/C_i,hi) C_i,q. The constant
    eta is unnecessary for the derivative and checked independently as residual.
    """
    q = system.state.positions.detach().clone().requires_grad_()
    h = system.state.supports.detach().clone().requires_grad_()
    s = replace(system.state, positions=q, supports=h)
    U = potential(s, cfg, sc)
    Uq, Uh = torch.autograd.grad(U, (q,h), retain_graph=True)
    if sc.adaptiveSupportCorrections:
        W, dW = cubic(s, cfg)
        n = W.sum(1); dim=q.shape[1]
        C = n*h.pow(dim)
        Ch = h.pow(dim)*(dW.sum(1)+dim*n/h)
        implicit, = torch.autograd.grad((C*(Uh/Ch).detach()).sum(), q)
        return U.detach(), Uq-implicit
    return U.detach(), Uq


def norm(value):return float(torch.linalg.vector_norm(value))


def operator_probe(dim=1,nx=24,adaptive=True):
    started=time.perf_counter();system,cfg,sc=fixture(dim,nx,adaptive)
    p,F=close(system,cfg,sc);U,g=independent_gradient(system,cfg,sc)
    scale=max(norm(F),1e-30)
    result=dict(dim=dim,particles=len(system.state.masses),adaptive=adaptive,
        supportThreshold=sc.adaptiveSupportThreshold,
        potential=float(U),productionPotential=float((system.state.masses*p.internalEnergies).sum()),
        forceGradientRelativeError=norm(F+g)/scale)
    if adaptive:
        h=system.state.supports;n=p.sums.numberDensity
        target=volumeToSupport(1/n,cfg.targetNeighbors,dim)
        result['supportConstraintRelativeResidual']=float(((h-target)/h).abs().max())
        warm=[]
        for factor in (.7,1.3):
            trial=clone(system);trial.state.supports=trial.state.supports*factor
            _,Ft=close(trial,cfg,sc)
            warm.append(dict(factor=factor,relativeForceDifference=norm(Ft-F)/scale,
                             relativeSupportDifference=float(((trial.state.supports-h)/h).abs().max())))
        result['warmStartProbes']=warm
    generator=torch.Generator(device=F.device).manual_seed(915+dim)
    directions=torch.randn((4,)+F.shape,device=F.device,dtype=F.dtype,generator=generator)
    directions=directions/torch.linalg.vector_norm(directions.flatten(1),dim=1)[:,None,None]
    spacing=2/nx;eps=spacing*2e-5
    derivatives=[];badDerivatives=[];finite=[]
    for d in directions:
        forces=[];badForces=[];energies=[]
        for sign in (-1,1):
            trial=clone(system);trial.state.positions=trial.state.positions+sign*eps*d
            pressure,force=close(trial,cfg,sc);forces.append(force)
            energies.append(float(potential(trial.state,cfg,sc)))
            bad=replace(pressure,gradH=torch.zeros_like(pressure.gradH))
            a,_=computePressureForce(trial.state,bad,cfg,trial.adjacency)
            badForces.append(trial.state.masses[:,None]*a)
        finite.append(dict(finiteDifference=(energies[1]-energies[0])/(2*eps),forceContraction=-float((F*d).sum())))
        derivatives.append((forces[1]-forces[0])/(2*eps))
        badDerivatives.append((badForces[1]-badForces[0])/(2*eps))
    D=directions.flatten(1);JF=torch.stack(derivatives).flatten(1)
    projected=D@JF.T
    result['projectedCanonicalForceJacobianSkew']=norm(projected-projected.T)/max(norm(projected),1e-30)
    result['directionalEnergyChecks']=finite
    result['directionalEnergyRelativeError']=max(abs(v['finiteDifference']-v['forceContraction'])for v in finite)/scale
    if adaptive:
        bad=replace(p,gradH=torch.zeros_like(p.gradH));a,_=computePressureForce(system.state,bad,cfg,system.adjacency)
        result['omittedGradHForceGradientRelativeError']=norm(system.state.masses[:,None]*a+g)/scale
        projected=D@torch.stack(badDerivatives).flatten(1).T
        result['omittedGradHJacobianSkew']=norm(projected-projected.T)/max(norm(projected),1e-30)
    result['wallSeconds']=time.perf_counter()-started
    return result


def structural_probe(adaptive=True):
    system,cfg,sc=fixture(adaptive=adaptive)
    base=pesphConservativeStep(clone(system),1e-3,cfg,sc)[0]
    generator=torch.Generator(device=system.state.positions.device).manual_seed(9)
    records=[]
    for variant in PESPHVariant:
        vs=[];driftErrors=[]
        for k in range(4):
            trial=clone(system)
            # A thermodynamic fields are q/A-derived; E stays authoritative when
            # changing velocity. Holding u fixed here would hide E's coupling.
            from warpSPH.systems.pesph import pesphStateType
            vals=vars(trial.state).copy();vals['thermodynamicVariable']='entropy'if variant is PESPHVariant.PressureEntropy else'energy'
            trial.state=pesphStateType(variant)(**vals)
            v=trial.state.velocities
            trial.state.velocities=v+(0 if k==0 else .03*torch.randn(v.shape,dtype=v.dtype,device=v.device,generator=generator))
            update,*_=pesphConservativeStep(trial,1e-3,cfg,replace(sc,variant=variant))
            driftErrors.append(norm(update.dxdt-trial.state.velocities))
            vs.append(update)
        records.append(dict(variant=variant.name,
            velocityPerturbationForceRelativeChange=max(norm(v.dvdt-vs[0].dvdt)for v in vs)/norm(vs[0].dvdt),
            extraIntegratedRate=max(norm(v.dAdt if variant is PESPHVariant.PressureEntropy else v.dEdt)for v in vs),
            driftVelocityError=max(driftErrors)))
    dtDifferences=[]
    for dt in (-.02,.0001,.02):
        update,*_=pesphConservativeStep(clone(system),dt,cfg,sc)
        dtDifferences.append(norm(update.dvdt-base.dvdt))
    verdict=classify_separable(pesphConservativeStep,clone(system),dt=.001,rhs_args=(cfg,sc))
    return dict(adaptive=adaptive,variants=records,accelerationDtDifferences=dtDifferences,
                installedADClassifier=asdict(verdict))


def map_probe(adaptive=True,omitted_grad_h=False):
    started=time.perf_counter();initial,cfg,sc=fixture(nx=12,adaptive=adaptive)
    close(initial,cfg,sc)
    method=get_scheme('Velocity Verlet');mass=initial.state.masses[:,None]
    z=torch.cat((initial.state.positions.flatten(),(mass*initial.state.velocities).flatten()))
    n=z.numel()//2
    def rhs(system,dt,*args,**kwargs):
        update,adj,state=pesphConservativeStep(system,dt,*args,**kwargs)
        if omitted_grad_h:
            pressure=computePressureState(state,cfg,sc,adj)
            pressure=replace(pressure,gradH=torch.zeros_like(pressure.gradH))
            update.dvdt=computePressureForce(state,pressure,cfg,adj)[0]
        return update,adj,state
    def step(value,dt=.004):
        system=clone(initial)
        system.state.positions=value[:n].reshape_as(system.state.positions)
        system.state.velocities=value[n:].reshape_as(system.state.velocities)/mass
        result=method(system,dt,rhs,cfg,sc).state.state
        return torch.cat((result.positions.flatten(),(mass*result.velocities).flatten()))
    eps=2e-6;J=torch.empty((2*n,2*n),dtype=z.dtype,device=z.device)
    for i in range(2*n):
        d=torch.zeros_like(z);d[i]=eps
        J[:,i]=(step(z+d)-step(z-d))/(2*eps)
    eye=torch.eye(n,dtype=z.dtype,device=z.device);zero=torch.zeros_like(eye)
    omega=torch.cat((torch.cat((zero,eye),1),torch.cat((-eye,zero),1)),0)
    reversed=step(step(z),-.004)
    return dict(adaptive=adaptive,omittedGradH=omitted_grad_h,
        canonicalSymplecticDefect=norm(J.T@omega@J-omega)/norm(omega),
        reversalLInf=float((reversed-z).abs().max()),wallSeconds=time.perf_counter()-started)


@pytest.mark.parametrize('dim,nx',[(1,24),(2,8),(3,6)])
@pytest.mark.parametrize('adaptive',[False,True])
def test_real_pressure_entropy_force_is_gradient_and_canonical_jacobian_symmetric(dim,nx,adaptive):
    r=operator_probe(dim,nx,adaptive)
    assert r['forceGradientRelativeError']<2e-10
    assert r['directionalEnergyRelativeError']<2e-7
    assert r['projectedCanonicalForceJacobianSkew']<2e-6
    assert abs(r['potential']-r['productionPotential'])<2e-12*abs(r['potential'])
    if adaptive:
        assert r['supportConstraintRelativeResidual']<2e-11
        assert max(v['relativeForceDifference']for v in r['warmStartProbes'])<2e-10
        assert r['omittedGradHForceGradientRelativeError']>1e-3
        assert r['omittedGradHJacobianSkew']>1e-4


@pytest.mark.parametrize('adaptive',[False,True])
def test_adiabatic_a_is_velocity_independent_but_authoritative_e_is_not(adaptive):
    r=structural_probe(adaptive)
    A=next(v for v in r['variants']if v['variant']=='PressureEntropy')
    E=next(v for v in r['variants']if v['variant']=='PressureEnergy')
    assert A['velocityPerturbationForceRelativeChange']==0
    assert A['extraIntegratedRate']==0
    assert E['velocityPerturbationForceRelativeChange']>1e-5
    assert E['extraIntegratedRate']>1e-5
    assert all(v['driftVelocityError']==0 for v in r['variants'])
    assert max(r['accelerationDtDifferences'])==0


@pytest.mark.parametrize('adaptive',[False,True])
def test_stock_verlet_canonical_map_and_reverse_step(adaptive):
    r=map_probe(adaptive)
    assert r['canonicalSymplecticDefect']<2e-7
    assert r['reversalLInf']<2e-11


def test_canonical_probe_rejects_the_omitted_grad_h_control():
    r=map_probe(True,True)
    assert r['canonicalSymplecticDefect']>1e-5
