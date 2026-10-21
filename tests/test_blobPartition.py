"""Wind-cloud benchmark states and cold-cloud mass diagnostic."""
import json
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from warpSPH.caseUtils.compressible.blob import sampleBlob,blobScales
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from study_blobPartition import Measures


def test_mass_matched_blob_source_geometry_and_thermodynamics():
    a=sampleBlob(256);cloud=a['materials']==1;r=np.linalg.norm(a['positions']-[5.,5.],axis=1)
    assert (r[cloud]<1).all() and (r[~cloud]>=1).all()
    assert len(a['masses'])>16384 and cloud.sum()>1200
    np.testing.assert_allclose(a['masses'][cloud][0]/a['masses'][~cloud][0],1.024,rtol=1e-6)
    np.testing.assert_allclose((5/3-1)*a['densities']*a['internalEnergies'],1.,rtol=2e-7)
    np.testing.assert_allclose(a['velocities'][cloud],0)
    np.testing.assert_allclose(a['velocities'][~cloud,0]/np.sqrt(5/3),2.7,rtol=1e-7)
    assert abs(blobScales()['tKH']-2.903)<.001


def test_equal_spacing_changes_mass_not_material_geometry():
    a=sampleBlob(128,massMatch=False);cloud=a['materials']==1
    assert len(a['masses'])==128*32
    np.testing.assert_allclose(a['supports'],a['supports'][0])
    np.testing.assert_allclose(a['masses'][cloud][0]/a['masses'][~cloud][0],10.)


def test_blob_mass_fraction_measures_original_cloud_and_both_thresholds():
    a=sampleBlob(64);measure=Measures(blobScales(),a)
    initial=measure(a,0.);assert initial['cloudMassFraction']==1 and initial['relativeEnergyDrift']==0
    cloud=np.flatnonzero(a['materials']==1);selected=cloud[:len(cloud)//2]
    a['internalEnergies'][selected]=2.
    row=measure(a,blobScales()['tKH'])
    assert .45<row['cloudMassFraction']<.55 and row['cloudDenseFraction']==1 and row['tKH']==1


def test_blob_runner_contract_is_available_with_energy_controls():
    from warpSPH.cases.compressible.blob import blobCase
    from warpSPH.runner import RunConfiguration
    spec=RunConfiguration('blob',solver={'crkDensityPolicy':'sameMaterial','crkEvolution':'spheral',
        'crkEnergyPartition':'spheral','crkEnergyPredictor':'raw'}).resolve(blobCase)
    assert spec.nx==256 and spec.dim==2 and spec.integrationScheme=='rungeKutta2'


@pytest.mark.parametrize('heating',[True,False])
@pytest.mark.parametrize('same_gamma',[True,False])
def test_published_entropy_fractions_use_the_common_pair_work_sign(heating,same_gamma):
    from types import SimpleNamespace as NS
    from warpSPH.modules.crk.energyControl import entropyRate
    adj=NS(paddedI=torch.tensor([0,1]),paddedJ=torch.tensor([1,0]),numNeighbors=torch.ones(2,dtype=torch.int64))
    v=torch.tensor([[1.],[-1.]],dtype=torch.float64)*(1 if heating else -1)
    a=torch.tensor([[-2.],[2.]],dtype=torch.float64);s=torch.tensor([.1,1.],dtype=torch.float64)
    gamma=torch.tensor([5/3,5/3 if same_gamma else 1.4],dtype=torch.float64)
    rate=entropyRate(adj,a,s,gamma,v,torch.zeros_like(v),0.)
    fraction=(1/1.1 if heating else .1/1.1) if same_gamma else .5
    np.testing.assert_allclose(float(rate[0]),(4 if heating else -4)*fraction,rtol=1e-13)
    np.testing.assert_allclose(float(rate.sum()),4 if heating else -4,rtol=1e-13)
