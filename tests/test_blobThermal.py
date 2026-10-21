"""Cloud diagnostics distinguish original material from the paper cold-gas cut."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from study_blobThermal import Measures


def initial():
    rho=np.array([1.,7.,10.,10.]);u=np.array([2.,1.,.15,.15]);g=5/3
    return dict(densities=rho,internalEnergies=u,pressures=(g-1)*rho*u,masses=np.ones(4),materials=np.array([0,0,1,1]),
        positions=np.array([[0.,5.],[2.,5.],[4.,5.],[6.,5.]]),velocities=np.array([[3.,0],[3.,0],[0.,0],[0.,0]]))


def test_paper_cold_mass_can_include_ambient_and_geometry_uses_original_cloud():
    a=initial();m=Measures(dict(tKH=2.,thermalGamma=5/3),a);r=m(a,0.)
    assert r['cloudMassFraction']==1.5 and r['originalCloudMassFraction']==1.
    assert r['cloudCOMx']==5 and r['cloudCOMy']==5 and r['cloudRMSRadius']==1.
    assert abs(r['entropyL1PerMass'])<1.e-15 and abs(r['cloudEntropyL1'])<1.e-15
    assert abs(r['relativeEnergyDrift'])<1.e-15


def test_log_entropy_measures_thermodynamics_and_cloud_shape_translates():
    a=initial();m=Measures(dict(tKH=2.,thermalGamma=5/3),a);b={k:v.copy() for k,v in a.items()}
    b['internalEnergies']*=2;b['pressures']*=2;b['positions']+=np.array([.5,-.25]);r=m(b,1.)
    np.testing.assert_allclose([r['entropyChangePerMass'],r['entropyL1PerMass'],r['cloudEntropyChange']],np.log(2.),rtol=2.e-15)
    assert r['cloudCOMx']==5.5 and r['cloudCOMy']==4.75 and r['cloudRMSRadius']==1.
