"""Local coefficients, unit conversion, dimensionality and actual Owen targets."""
from pathlib import Path
import numpy as np
import pytest
import torch
from warpSPHCore import KernelFunctions,get_torch_precision
from warpSPH.modules.adaptiveSupport.owenHermite import (coefficients,derivative_knots,
    monotonic_slopes,hermite_values,shell_values,HermiteOwen)


@pytest.mark.parametrize('dim',[1,2,3])
@pytest.mark.parametrize('kernel',[KernelFunctions.CubicSpline,KernelFunctions.QuarticSpline,
    KernelFunctions.QuinticSpline,KernelFunctions.B7,KernelFunctions.B8,KernelFunctions.Wendland2,
    KernelFunctions.Wendland4,KernelFunctions.Wendland6,KernelFunctions.Poly6,KernelFunctions.Spiky,
    KernelFunctions.HOCT4,KernelFunctions.Gaussian])
def test_kernel_specific_tables_are_finite_monotonic_and_invert_shells(kernel,dim):
    a=coefficients(kernel,dim);m=a['lookupMeta'];n=int(m[3]);v=a['lookupData']
    z=np.linspace(m[0],m[1],10000);y=hermite_values(z,m[0],m[1],v[:n],v[n:])
    assert np.isfinite(y).all() and np.all(np.diff(y)>=-1.e-12)
    nh=np.array([3.,4.,6.,8.,12.]);moment=shell_values(nh,dim,a['gradData'])
    recovered=hermite_values(np.log(moment+float(a['inverseLogOffset'])) if a['inverseLog'] else moment,m[0],m[1],v[:n],v[n:])
    np.testing.assert_allclose(recovered,nh,rtol=.005,atol=0.)
    # Endpoint tangent extrapolation must be continuous at both ends.
    eps=1.e-10*m[2]
    out=hermite_values(np.array([m[0]-eps,m[0],m[1],m[1]+eps]),m[0],m[1],v[:n],v[n:])
    assert abs(out[1]-out[0])<1.e-6 and abs(out[3]-out[2])<1.e-6


def test_linear_derivative_fit_and_monotonic_limiting_are_independent():
    x=np.linspace(0,3,101);y=2+4*x
    np.testing.assert_allclose(derivative_knots(y,x[1]-x[0]),4,rtol=2.e-14)
    np.testing.assert_allclose(monotonic_slopes(y,derivative_knots(y,.03),.03),4,rtol=2.e-14)


def test_local_b8_inverse_matches_exported_native_coefficients_after_unit_conversion():
    if get_torch_precision()!=torch.float64:pytest.skip('float64 reference')
    path=Path(__file__).resolve().parents[1]/'reports/materialEOS/idealH/operators/tables.npz'
    if not path.exists():pytest.skip('retained compiled native audit coefficients not available')
    a=coefficients(KernelFunctions.B8,2,points=1000,inverse_space='linear');native=dict(np.load(path));m=native['lookupMeta'];n=int(m[3])
    # R=4h; z_R=4^((d+1)/d)*z_native, n_R=4*nPerh.
    mlocal=a['lookupMeta'];np.testing.assert_allclose(mlocal[:3],m[:3]*8,rtol=2.e-9,atol=1.e-8)
    x=np.linspace(max(mlocal[0],8*m[0]),min(mlocal[1],8*m[1]),5000)
    expected=4*hermite_values(x/8,m[0],m[1],native['lookupData'][:n],native['lookupData'][n:])
    nlocal=int(mlocal[3]);actual=hermite_values(x,mlocal[0],mlocal[1],a['lookupData'][:nlocal],a['lookupData'][nlocal:])
    np.testing.assert_allclose(actual,expected,rtol=3.e-8,atol=1.e-9)


@pytest.mark.parametrize('dim',[1,2,3])
@pytest.mark.parametrize('weighted',[False,True])
def test_full_generated_support_path_needs_no_material_table_when_unweighted(dim,weighted):
    from types import SimpleNamespace as NS
    from warpSPHCore import ParticleState,DomainDescription,SupportScheme
    from warpSPH.configurations import CRKSPHConfig
    from warpSPH.modules.adaptiveSupport.optimalSupportOwen import evaluateOptimalSupportOwen
    dtype=get_torch_precision();T=lambda a:torch.tensor(a,dtype=dtype)
    x=torch.zeros((2,dim),dtype=dtype);x[1,0]=.2
    state=ParticleState(x,T([1.,1.]),T([1.,1.]),torch.zeros(2,dtype=torch.int32),T([1.,1.]))
    if weighted:state.materials=torch.tensor([0,1],dtype=torch.int32);state.masses[1]=2.
    domain=DomainDescription(T([-2.]*dim),T([2.]*dim),torch.zeros(dim,dtype=torch.bool),dim)
    sphere=(2.,np.pi,4*np.pi/3)[dim-1]
    c=NS(kernel=KernelFunctions.B8,domain=domain,targetNeighbors=sphere*4**dim,verletScale=1.4,supportMode=SupportScheme.KernelMeanSymmetric)
    sc=CRKSPHConfig(owenTable='hermiteShell',owenMaterialWeights=weighted,adaptiveSupportIterations=1)
    _,support,_,moments,_=evaluateOptimalSupportOwen(state,c,sc,validateOnce=True,computeDensity=False)
    inferred=HermiteOwen(KernelFunctions.B8,dim).fromPsiH(moments[-1]);s=4/inferred
    alpha=torch.where(s>=1,.4*(1+s**-3),.4*(1+s*s));expected=(1-alpha+alpha*s).clamp(.25,4.)
    tol=1.e-14 if dtype==torch.float64 else 2.e-6
    torch.testing.assert_close(support,expected,rtol=tol,atol=tol)


def test_generated_mode_config_roundtrip_and_bad_kernel_fail_explicitly():
    from warpSPH.configurations import CRKSPHConfig
    from warpSPH.configurations.crkSPH import crkSPHConfigToDict,dictToCRKSPHConfig
    c=CRKSPHConfig(owenTable='hermiteShell');c.validate()
    assert dictToCRKSPHConfig(crkSPHConfigToDict(c)).owenTable=='hermiteShell'
    with pytest.raises(ValueError,match='hydrodynamic'):coefficients(KernelFunctions.AdhesionKernel,2)
    c.owenTable='typo'
    with pytest.raises(ValueError,match='owenTable'):c.validate()


def test_shifted_log_preserves_zero_lower_endpoint():
    from warpSPH.modules.adaptiveSupport.owenHermite import build_coefficients
    # A compact gradient supported only at q<.9 gives zero moment at nh=1.1.
    q=np.linspace(0,1,1001);gradient=-np.where(q<.9,(.9-q)**3,0.)
    a=build_coefficients(gradient,2);m=a['lookupMeta'];n=int(m[3]);y=a['lookupData']
    assert a['inverseLogOffset']>0 and np.isfinite(y).all()
    np.testing.assert_allclose(hermite_values(m[0],m[0],m[1],y[:n],y[n:]),1.1,atol=1.e-14)


@pytest.mark.parametrize('scheme',['Monaghan','CompSPH','CRKSPH'])
@pytest.mark.parametrize('table',['lattice','shell','hermiteShell'])
def test_runner_accepts_and_preserves_owen_controls(scheme,table,tmp_path):
    from warpSPH.runner import RunConfiguration,resolvedConfiguration,configurationSchema
    from warpSPH.cases.compressible.sod import sodCase
    document=RunConfiguration('sod',params=dict(viscositySwitch='NoneSwitch'),solver=dict(scheme=scheme,owenTable=table,owenMaterialWeights=True,
        materials=[dict(name='gas',eos=dict(type='idealGas',gamma=1.4))]))
    loaded=RunConfiguration.load(document.save(tmp_path/'owen.json'))
    ctx=loaded.prepare(sodCase)
    assert ctx.schemeConfig.owenTable==table and ctx.schemeConfig.owenMaterialWeights
    snapshot=resolvedConfiguration(ctx)['solverConfig']
    assert snapshot['owenTable']==table and snapshot['owenMaterialWeights']
    schema=configurationSchema(sodCase,scheme)['properties']['solver']['properties']
    assert table in schema['owenTable']['enum']
    assert schema['owenMaterialWeights']['type']=='boolean'


@pytest.mark.parametrize('values,match',[
    (dict(owenTable='typo'),'owenTable'),
    (dict(owenMaterialWeights='true'),'owenMaterialWeights'),
    (dict(owenMaterialWeights=True),'materials table'),
])
def test_runner_rejects_invalid_owen_configuration(values,match):
    from warpSPH.runner import RunConfiguration
    from warpSPH.cases.compressible.sod import sodCase
    with pytest.raises(ValueError,match=match):
        RunConfiguration('sod',solver=dict(scheme='CRKSPH',**values)).prepare(sodCase)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA stream ordering')
def test_cold_lattice_lookup_generation_uses_current_torch_stream(monkeypatch):
    import warp as wp
    from warpSPH.modules.adaptiveSupport.wp_psi import generatePSILut_warp,computePsi
    from warpSPHCore.autograd.stream import torchWarpStream
    wp.init()
    device=torch.device('cuda',torch.cuda.current_device())
    with torchWarpStream(device):
        reference=generatePSILut_warp(KernelFunctions.Wendland2,2.,6.,65)
    torch.cuda.synchronize(device)
    launch=wp.launch
    observed=[]
    def checked(kernel,*args,**kwargs):
        if kernel==computePsi:
            selected=kwargs.get('stream') or wp.get_stream(wp.device_from_torch(device))
            # The consumer copies use Torch's current stream. A different
            # producer stream needs an explicit dependency; this generator
            # promises the current-stream contract, without a global fence.
            assert selected.cuda_stream==torch.cuda.current_stream(device).cuda_stream
            observed.append(1)
        return launch(kernel,*args,**kwargs)
    monkeypatch.setattr(wp,'launch',checked)
    stream=torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        actual=generatePSILut_warp(KernelFunctions.Wendland2,2.,6.,65)
    stream.synchronize()
    assert len(observed)==3
    for a,b in zip(actual,reference):torch.testing.assert_close(a,b,rtol=0,atol=0)
