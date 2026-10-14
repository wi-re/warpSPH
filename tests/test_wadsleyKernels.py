"""The warp kernels of the Wadsley detector (`modules/shockCapturing/Wadsley2017/kernels.py`) against the torch operations they replace:
the whole detector (`wadsleyDetector` vs `wadsleyDetectorReference`), which must also be free of host syncs. (The pseudo-inverse
inside it is `warpSPH.linalg`, tested in `test_linalg.py`.)"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='warp kernels')

from warpSPH.modules.shockCapturing.Wadsley2017 import wadsleyDetector, wadsleyDetectorReference
from warpSPH.runner import run
from warpSPHCore import get_torch_precision


def _state():
    from warpSPH.cases.compressible.greshoVortex import greshoVortexCase
    res = run(greshoVortexCase, scheme='Monaghan', nx=40, nSteps=2, progress=False, quiet=True)
    st = res.state.state
    x = st.positions
    st.pressures = 1.0 + 0.2 * x[:, 0] + 0.05 * torch.sin(6.0 * x[:, 1])
    st.velocities = torch.stack([-0.7 * x[:, 0] + 0.3 * torch.sin(5.0 * x[:, 1]), 0.4 * x[:, 0] * x[:, 1]], 1).to(x.dtype)
    return res.state, res.ctx.config, res.ctx.schemeConfig


def test_detectorMatchesTheTorchReferenceAndDoesNotSync():
    system, config, schemeConfig = _state()
    st = system.state
    ref = wadsleyDetectorReference(st, config, system.adjacency, schemeConfig.gamma)
    new = wadsleyDetector(st, config, system.adjacency, schemeConfig.gamma)
    for name, a, b in zip(('D', 'Tnorm', 'R', 'n'), ref, new):
        scale = a.abs().max().clamp(min=1e-30)
        assert ((a - b).abs().max() / scale) < 2e-3, name
    assert (new[2].abs() <= 1.0).all()
    torch.cuda.set_sync_debug_mode('error')
    try:
        wadsleyDetector(st, config, system.adjacency, schemeConfig.gamma)
    finally:
        torch.cuda.set_sync_debug_mode('default')
