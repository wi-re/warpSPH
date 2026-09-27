"""CRKSPH warns (once per support mode) when run with a support mode other
than the kernel-mean pair kernel it is formulated with -- under e.g. Gather
it silently stops conserving total energy (Sod nx=200: -7.3e-6)."""

import warnings
from types import SimpleNamespace

import pytest
from warpSPHCore import SupportScheme

from warpSPH.schemes import crkSPH


@pytest.fixture(autouse=True)
def _fresh():
    crkSPH._warnedSupportModes.clear()
    yield
    crkSPH._warnedSupportModes.clear()


def test_warns_once_for_non_symmetric_support():
    cfg = SimpleNamespace(supportMode=SupportScheme.Gather)
    with pytest.warns(RuntimeWarning, match="supportMode=Gather"):
        crkSPH._warnNonConservativeSupport(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # second call: silent
        crkSPH._warnNonConservativeSupport(cfg)


def test_silent_for_kernel_mean_symmetric():
    cfg = SimpleNamespace(supportMode=SupportScheme.KernelMeanSymmetric)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        crkSPH._warnNonConservativeSupport(cfg)


def test_visible_despite_blanket_ignore_filter():
    # a blanket warnings.filterwarnings("ignore") (as the SDF module used to
    # install at import) must not hide the CRK warning
    cfg = SimpleNamespace(supportMode=SupportScheme.Scatter)
    with warnings.catch_warnings(record=True) as rec:
        warnings.filterwarnings("ignore")        # what that module does
        crkSPH._warnNonConservativeSupport(cfg)
    assert any(issubclass(r.category, crkSPH.CRKSupportWarning) for r in rec)
