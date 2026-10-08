"""Bootstrap the runtime before anything imports ``warpSPH``.

pytest imports ``conftest.py`` before collecting test modules, which is the only
place precision can still be chosen -- see :mod:`warpSPHBootstrap`.
"""

import os

import pytest

from warpSPHBootstrap import bootstrap

# float32 unless the environment asks otherwise (tests/test_reconstruction.py reruns itself in float64 this way)
RUNTIME = bootstrap(precision=os.environ.get('warpSPHCore_PRECISION', 'float32'))


@pytest.fixture(scope='session')
def runtime():
    return RUNTIME


@pytest.fixture(scope='session')
def exportRoot(tmp_path_factory):
    """Keep any test that stores output out of the repo's `export/` tree."""
    return str(tmp_path_factory.mktemp('export'))
