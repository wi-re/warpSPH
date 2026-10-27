"""Periodic 2D wind-cloud blob, Frontiere et al. 2017 section 4.5.1."""
import torch
from ...configurations import EOSConfig,MaterialConfig
from ...configurations.pesph import PESPHConfig
from ...runner import Case,registerCase,caseMain
from ...runner.parameters import Parameter as P
from ...caseUtils.compressible.blob import sampleBlob,blobScales
from ...caseUtils.compressible.materialValidation import buildMaterialSystem,materialConservation,hostScalars
from ..common.parameters import COMPRESSIBLE_CASE_PARAMETERS
from ..common.plotting import Field,particlePlot
from .common import COMPRESSIBLE_DEFAULTS,COMPRESSIBLE_PARAMS,configureCompressible,compressibleTimestep,paramExtraData


def configureScheme(ctx):
    if ctx.param('rho0')!=1.:raise ValueError('blob nondimensional ambient density is fixed at one')
    configureCompressible(ctx)
    if not isinstance(ctx.schemeConfig, PESPHConfig):
        ctx.schemeConfig.materials=[MaterialConfig(name,EOSConfig(gamma=ctx.param('gamma'))) for name in ('ambient','cloud')]
    ctx.config.domain.min=torch.zeros(2,device=ctx.device,dtype=ctx.dtype)
    ctx.config.domain.max=torch.tensor([40.,10.],device=ctx.device,dtype=ctx.dtype)*ctx.spec.L


def buildSystem(ctx):
    if ctx.spec.dim!=2 or not bool(ctx.config.domain.periodic.all()):raise ValueError('blob requires 2D periodic domain')
    a=sampleBlob(ctx.spec.nx,ctx.param('massMatch'),ctx.spec.n_h,ctx.param('gamma'),ctx.param('mach'),ctx.param('contrast'),ctx.spec.L)
    tensor=lambda key:torch.as_tensor(a[key],device=ctx.device,dtype=torch.int32 if key=='materials' else ctx.dtype)
    return buildMaterialSystem(ctx,tensor('positions'),tensor('masses')/tensor('densities'),tensor('densities'),tensor('pressures'),tensor('materials'),tensor('velocities'))


def diagnostics(ctx,system):
    s=system.state;cloud=s.materials==1;mass=torch.where(cloud,s.masses,0.)
    cold=(s.densities>.64*ctx.param('contrast'))&(s.internalEnergies<.9/(ctx.param('gamma')-1))
    scales=blobScales(ctx.param('gamma'),ctx.param('mach'),ctx.param('contrast'))
    return hostScalars(dict(materialConservation(ctx,system),
        tKH=float(system.t)/(scales['tKH']*ctx.spec.L),cloudMassFraction=torch.where(cold,mass,0.).sum()/mass.sum(),
        cloudDenseFraction=torch.where(s.densities>.64*ctx.param('contrast'),mass,0.).sum()/mass.sum(),
        cloudMeanEnergy=(mass*s.internalEnergies).sum()/mass.sum()))


def postStep(ctx,system,step):
    # Keep the full periodic wind visible in the daemon and local video over
    # multiple box crossings. This is a coordinate wrap, not a reservoir BC.
    s=system.state;lo=ctx.config.domain.min;length=ctx.config.domain.max-lo
    s.positions=lo+torch.remainder(s.positions-lo,length)


setupPlot,updatePlot=particlePlot([Field('densities','Density',vMin=.5,vMax=10.),
    Field('materials','Ambient / original cloud',vMin=0,vMax=1),Field('pressures','Pressure',vMin=.5,vMax=12.)],figsize=(15,3.5))
blobCase=registerCase(Case(name='blob',scheme='CRKSPH',
    description='2D periodic Mach-2.7 wind / density-10 cloud; published CRKSPH geometry.',
    configureScheme=configureScheme,buildSystem=buildSystem,diagnostics=diagnostics,timestep=compressibleTimestep,postStep=postStep,
    setupPlot=setupPlot,updatePlot=updatePlot,extraData=paramExtraData,extraFields=('internalEnergies','supports'),supportedSchemes=('CRKSPH','PESPH'),
    defaults=dict(COMPRESSIBLE_DEFAULTS,caseName='blob',dim=2,nx=256,L=1.,kernel='QuinticSpline',n_h=4.05,tLimit=8*blobScales()['tKH'],cflFactor=.15,plotInterval=100),
    params=dict(COMPRESSIBLE_PARAMS,viscositySwitch='NoneSwitch',massMatch=True,mach=2.7,contrast=10.,initialDensity='prescribed'),
    parameterDefinitions={**COMPRESSIBLE_CASE_PARAMETERS,
        'massMatch':P('boolean','Near mass-matched cloud lattice, rounded as in Spheral; otherwise equal spacing.','sampling'),
        'mach':P('number','Ambient wind Mach number.','initial',exclusiveMinimum=0),
        'contrast':P('number','Cloud/ambient density ratio.','initial',exclusiveMinimum=1),
        'initialDensity':P('string','Density used for initialization energy.','initial',choices=('prescribed','summed'))},
    simulationParameters={'dim':P('integer','Two-dimensional wind-cloud case.','domain',choices=(2,))},
    variants={'smoke':{'simulation':{'nx':64,'nSteps':5,'tLimit':.1}},'equalSpacing':{'params':{'massMatch':False}}}))

if __name__=='__main__':caseMain(blobCase)
