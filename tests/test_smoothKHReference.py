"""Check common ICs and the mode diagnostic against the pinned native driver."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_spheralKH import initial_fields,sample,mixing_mode,Measures


def native_source():
    path=Path(__file__).resolve().parents[1]/'results/materialEOS/spheral-build/source/tests/functional/Hydro/KelvinHelmholtz/KelvinHelmholtz-2d_McNally.py'
    if not path.exists():pytest.skip('Pinned Spheral source not installed')
    return path,ast.parse(path.read_text())


def test_single_material_pressure_balance_and_native_mode_normalization():
    a=sample(64)
    np.testing.assert_allclose(a['pressures'],(5/3-1)*a['densities']*a['internalEnergies'],atol=1.e-15)
    np.testing.assert_allclose(a['masses']/a['densities'],1/64**2,atol=1.e-18)
    np.testing.assert_allclose(mixing_mode(a),.01,atol=1.e-16)
    assert np.unique(a['materials']).tolist()==[0]
    metrics=Measures({},a)(a,0.)
    for key in ('momentumError','relativeEnergyDrift','entropyChangePerMass'):assert metrics[key]==0


def test_common_fields_against_actual_native_initialization_statements():
    path,tree=native_source()
    # Execute only the physical four-region assignments in the one-NodeList
    # initialization, with tiny field-list doubles; never import its simulation.
    branch=next(n for n in tree.body if isinstance(n,ast.If) and ast.unparse(n.test)=='numNodeLists == 2' and n.lineno>300)
    loop=next(n for n in branch.orelse if isinstance(n,ast.For))
    pos=np.array([[.037,.0],[.12,.249],[.22,.25],[.32,.4],[.42,.5],[.52,.749],[.62,.75],[.72,.999]])
    points=[SimpleNamespace(x=x,y=y) for x,y in pos];rho=np.zeros(len(pos));vel=[None]*len(pos);eps=np.zeros(len(pos))
    mass=np.where((pos[:,1]<.25)|(pos[:,1]>=.75),1.,2.)
    scope=dict(nodes1=SimpleNamespace(numInternalNodes=len(pos)),pos=points,rho=rho,vel=vel,eps=eps,mass=mass,
        rho1=1.,rho2=2.,rhom=-.5,vx1=.5,vx2=-.5,vxm=.5,delta=.01,smooth=.025,
        sin=np.sin,exp=np.exp,pi=np.pi,Vector=lambda *v:np.array(v),P1=2.5,gamma=5/3,vxboost=0.,vyboost=0.)
    exec(compile(ast.Module(body=[loop],type_ignores=[]),str(path),'exec'),scope)
    r,v=initial_fields(pos)
    np.testing.assert_allclose(r,rho,rtol=0,atol=1.e-15);np.testing.assert_allclose(v,vel,rtol=0,atol=1.e-15)
    np.testing.assert_allclose(mass,rho,rtol=0,atol=1.e-15)
    np.testing.assert_allclose(eps,2.5/((5/3-1)*r),rtol=0,atol=1.e-15)


def test_mode_diagnostic_against_actual_native_function_on_irregular_state(tmp_path):
    import os
    path,tree=native_source();function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='mixingScale')
    a=sample(12);a['positions']=(a['positions']+.017*np.sin(np.arange(144)[:,None]+[0.,1.]))%1
    x=a['positions'][:,0];a['supports']*=1+.1*np.cos(2*np.pi*x);a['velocities'][:,1]+=.02*np.cos(4*np.pi*x)
    def field(values):return SimpleNamespace(internalValues=lambda:values)
    nodes=SimpleNamespace(positions=lambda:field([SimpleNamespace(x=x,y=y) for x,y in a['positions']]),
        velocity=lambda:field([SimpleNamespace(x=x,y=y) for x,y in a['velocities']]),
        massDensity=lambda:field(a['densities']),
        Hfield=lambda:field([SimpleNamespace(Determinant=lambda R=R:(4/R)**2) for R in a['supports']]))
    scope=dict(mpi=SimpleNamespace(rank=0,SUM=None,reduce=lambda values,op:values),nodeSet=[nodes],
        sqrt=np.sqrt,sin=np.sin,cos=np.cos,exp=np.exp,pi=np.pi,os=os,dataDir=str(tmp_path),mixFile='mode.txt')
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),scope)
    scope['mixingScale'](0,.3,.001);_,mode,ke=np.loadtxt(tmp_path/'mode.txt')
    np.testing.assert_allclose(mixing_mode(a),mode,rtol=2.e-14)
    np.testing.assert_allclose(ke,(.5*a['densities']*a['velocities'][:,1]**2).max(),rtol=1.e-15)
