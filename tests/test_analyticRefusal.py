"""Analytic walls are hooked into the weakly compressible delta+ scheme only: every other scheme must refuse them loudly (before the guard they ran with no wall at
all, the fluid simply fell through the open box)."""
import pytest
import torch

pytest.importorskip('warpSPHBoundaries')

from warpSPH.cases import importAll  # noqa: E402
from warpSPH.runner import getCase, run  # noqa: E402
from warpSPH.runner.caseSpec import CaseSpec  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')


@pytest.mark.parametrize('scheme', ['divergenceFree', 'omniIncompressible', 'artificialCompressible'])
def test_other_schemes_refuse_analytic_walls(scheme):
    importAll()
    case = getCase('dambreak')
    spec = CaseSpec(caseName='refusal', scheme=scheme, params={**case.params, 'wallRepresentation': 'analytic'}).merged(**case.defaults).merged(
        scheme=scheme, nx=24, nSteps=2, plot=False, store=False, progress=False, video=False, show=False, quiet=True)
    with pytest.raises(NotImplementedError, match='weakly compressible delta'):
        run(case, spec)
