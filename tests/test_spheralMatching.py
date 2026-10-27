"""Reference fraction semantics before full matched trajectory comparisons."""
import ctypes
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from study_spheralMatching import reference_fraction, matching_options


def test_matching_context_preserves_both_density_and_energy_controls():
    source=dict(schedule='paper',matchingEnergyPartition='entropy',matrixControls=dict(solver=dict(untouched=42)))
    options=matching_options(source)
    assert options['matrixControls']['solver']==dict(untouched=42,crkEvolution='paper',crkEnergyPartition='entropy',crkEnergyPredictor='raw')
    assert source['matrixControls']['solver']==dict(untouched=42)


def test_reference_fraction_thresholds_and_pair_complement():
    si=torch.tensor([1.,1.,1.,1.,1.],dtype=torch.float64)
    sj=torch.tensor([3.,3.,3.,1.+4.e-16,3.],dtype=torch.float64)
    work=torch.tensor([1.,-1.,1.e-15,1.,-1.e-16],dtype=torch.float64)
    f=reference_fraction(si,sj,work)
    torch.testing.assert_close(f,torch.tensor([.75,.25,.5,.5,.5],dtype=torch.float64),rtol=0,atol=0)
    torch.testing.assert_close(f+reference_fraction(sj,si,work),torch.ones_like(f),rtol=0,atol=0)


def test_reference_fraction_against_actual_compiled_native_function():
    manifest=Path('results/materialEOS/spheral-heating-policies/build.json')
    if not manifest.exists():pytest.skip('Pinned native component library is not installed')
    import json
    build=json.loads(manifest.read_text())
    library=Path(build.get('libraryPath',manifest.parent/'libSpheralPaperPolicies.so'))
    native=ctypes.CDLL(str(library.resolve())).warpsph_paper_entropy_fraction
    native.argtypes=[ctypes.c_double]*3;native.restype=ctypes.c_double
    rng=np.random.default_rng(683)
    si=np.exp(rng.uniform(-8,8,1000));sj=np.exp(rng.uniform(-8,8,1000));work=rng.uniform(-1,1,1000)
    si=np.r_[si,1.,1.,1.];sj=np.r_[sj,3.,1.+4.e-16,3.];work=np.r_[work,1.e-15,1.,-1.e-16]
    expected=np.array([native(float(a),float(b),float(w)) for a,b,w in zip(si,sj,work)])
    actual=reference_fraction(*(torch.tensor(x) for x in (si,sj,work))).numpy()
    np.testing.assert_allclose(actual,expected,rtol=0,atol=2.e-16)
