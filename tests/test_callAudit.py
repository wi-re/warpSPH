"""The opt-in call audit (`warpSPHCore/callAudit.py`): counts torch ops, warp launches and host syncs per
named section, off by default."""
import pytest
import torch
import warp as wp

from warpSPHCore import callAudit
from warpSPHCore.profiling import record_function

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')


@wp.kernel
def _noop(a: wp.array(dtype=wp.float32)):
    i = wp.tid()
    a[i] = a[i] + 1.0


@pytest.fixture
def audit():
    callAudit.reset()
    callAudit.enable()
    yield callAudit
    callAudit.disable()
    callAudit.reset()


def test_isOffByDefaultAndFree():
    assert not callAudit.enabled
    with record_function('x'):          # a nullcontext: nothing recorded, nothing installed
        torch.ones(3, device='cuda').sum()
    assert not callAudit._sections


def test_countsOpsLaunchesAndSyncsPerSectionWithTheCallSite(audit):
    x = torch.ones(8, device='cuda')
    arr = wp.from_torch(x)
    with record_function('outer'):
        y = x + 1
        with record_function('inner'):
            wp.launch(_noop, dim=8, inputs=[arr], device='cuda')
            wp.launch(_noop, dim=8, inputs=[arr], device='cuda')
            y.sum().item()                      # one implicit sync
        torch.cuda.synchronize()                # one explicit sync
    s = callAudit._sections
    assert s['inner'].launches == 2 and s['inner'].syncs == 1 and s['inner'].ops >= 2
    assert s['outer'].launches == 2 and s['outer'].syncs == 2           # inclusive
    assert s['outer'].selfSyncs == 1 and s['outer'].selfLaunches == 0   # excluding `inner`
    assert any('test_callAudit.py' in site for site in s['inner'].syncSites)
    text = callAudit.report()
    assert 'inner' in text and 'test_callAudit.py' in text


def test_stepReportNamesTheStepsThatSpike(audit):
    x = torch.ones(4, device='cuda')
    for step in range(3):
        with record_function('work'):
            for _ in range(step):
                x.sum().item()
        callAudit.markStep()
    assert 'step 2' in callAudit.stepReport(n=1)


def test_disableRestoresEverything():
    callAudit.enable()
    callAudit.disable()
    assert wp.launch.__name__ == 'launch' and torch.cuda.synchronize.__name__ == 'synchronize'
    assert not callAudit.enabled
