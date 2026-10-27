"""Pair-Q parity, backward defaults and cached midpoint timing controls."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch
import warp as wp
from warpSPHCore import scalar_t, domainData, KernelFunctions
from warpSPH.configurations.crkSPH import (CRKSPHConfig, CRKViscosity,
    crkSPHConfigToDict, dictToCRKSPHConfig, resolveCRKLimiter)
from warpSPH.configurations import MaterialConfig
from warpSPH.modules.compSPH.pairWork import beginPairWork
from warpSPH.modules.crk.viscosityControl import viscosityGradient
from warpSPH.modules.dissipation import DiffusionParameters, computeFrontiereQ


@wp.kernel
def sample_q(domain:domainData, params:DiffusionParameters, out:wp.array(dtype=scalar_t)):
    side=wp.tid()
    xi=wp.vec2(scalar_t(.4),scalar_t(.1));xj=wp.vec2(scalar_t(.1),scalar_t(.1))
    dv=wp.vec2(scalar_t(-1.2),scalar_t(.2))
    out[side*3]=computeFrontiereQ(xi,xj,scalar_t(.7),scalar_t(.5),scalar_t(1.),scalar_t(.182),dv,domain,
        wp.static(KernelFunctions.CubicSpline.value),scalar_t(2.),scalar_t(1.3),scalar_t(1.),scalar_t(1.),params,side==1)
    out[side*3+1]=computeFrontiereQ(xi,xj,scalar_t(.7),scalar_t(.5),scalar_t(1.),scalar_t(.182),dv,domain,
        wp.static(KernelFunctions.CubicSpline.value),scalar_t(2.),scalar_t(1.3),scalar_t(1.),scalar_t(1.),params,side==1,scalar_t(1.778002))
    out[side*3+2]=computeFrontiereQ(xi,xj,scalar_t(.7),scalar_t(.5),scalar_t(1.),scalar_t(.182),dv,domain,
        wp.static(KernelFunctions.CubicSpline.value),scalar_t(2.),scalar_t(1.3),scalar_t(1.),scalar_t(1.),params,side==1,scalar_t(2.))


def test_actual_pair_q_uses_requested_length_and_preserves_native_default():
    domain=domainData();domain.dim=2
    domain.domainMin=wp.array([0.,0.],dtype=scalar_t,device='cpu')
    domain.domainMax=wp.array([2.,2.],dtype=scalar_t,device='cpu')
    domain.periodicity=wp.array([False,False],dtype=wp.bool,device='cpu')
    params=DiffusionParameters();params.C_l=1.;params.C_q=.75
    out=wp.empty(6,dtype=scalar_t,device='cpu')
    wp.launch(sample_q,2,inputs=[domain,params,out],device='cpu');values=out.numpy().reshape(2,3)
    np.testing.assert_array_equal(values[:,0],values[:,1])
    for side,(R,rho,c) in enumerate(((.7,1.,2.),(.5,.182,1.3))):
        for k,scale in enumerate((1.778002,1.778002,2.)):
            h=R/scale;mu=h*(-1.2*.3)/(.3**2+.01*h*h)
            np.testing.assert_allclose(values[side,k],rho*(-c*mu+.75*mu*mu),rtol=2.e-6)


def test_cached_gradient_moves_between_stages_and_replay_starts_from_initial_cache():
    gprev=torch.tensor([[-1.]])
    work=beginPairWork([0.,1.]);work.crkQGradient=gprev
    stage0=NS(pairWork=work,state=NS(crkQGradient=gprev))
    g0=torch.tensor([[2.]]);g1=torch.tensor([[3.]])
    assert viscosityGradient(stage0,g0,'cached') is gprev
    stage1=NS(pairWork=work,state=NS(crkQGradient=gprev))
    assert viscosityGradient(stage1,g1,'cached') is g0
    assert stage1.state.crkQGradient is g1
    # New midpoint attempt is seeded from the unchanged initial state, not
    # the holder mutated by the preceding (discarded) attempt.
    replay=NS(pairWork=beginPairWork([0.,1.]),state=NS(crkQGradient=gprev))
    replay.pairWork.crkQGradient=replay.state.crkQGradient
    assert viscosityGradient(replay,g0,'cached') is gprev
    assert viscosityGradient(stage1,g0,'current') is g0
    assert stage1.state.crkQGradient is g1


def test_control_config_roundtrip_and_old_serialization_defaults():
    import json
    from warpSPH.runner.configuration import _validateSolver
    # Unassigned struct defaults can be NumPy scalars; the runner exports
    # these too, without first going through a solver override.
    json.dumps(crkSPHConfigToDict(CRKSPHConfig()))
    assert _validateSolver('CRKSPH', {'crkViscosityGradient':'cached',
        'crkViscosityParams':{'viscosityKernelScale':2.}})=={
        'crkViscosityGradient':'cached','crkViscosityParams.viscosityKernelScale':2.}
    c=CRKSPHConfig(crkEvolution='spheral',crkDensityPolicy='sameMaterial',materials=[MaterialConfig()],crkViscosityGradient='cached')
    c.crkViscosityParams.viscosityKernelScale=2.;c.validate()
    d=crkSPHConfigToDict(c);b=dictToCRKSPHConfig(d)
    assert b.crkViscosityGradient=='cached' and b.crkViscosityParams.viscosityKernelScale==2.
    assert resolveCRKLimiter(b.crkViscosityParams,3.02).viscosityKernelScale==2.
    del d['crkViscosityGradient'];del d['crkViscosityParams']['viscosityKernelScale']
    b=dictToCRKSPHConfig(d)
    assert b.crkViscosityGradient=='current' and b.crkViscosityParams.viscosityKernelScale==0.
    c.crkViscosityGradient='invalid'
    with pytest.raises(ValueError,match='crkViscosityGradient'):c.validate()
