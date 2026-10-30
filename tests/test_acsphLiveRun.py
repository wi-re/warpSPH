"""Native publication must preserve selected runner/hooks and visible failures."""
from types import SimpleNamespace as NS
import json
import threading
import numpy as np
import pytest
from scripts.acsph_reference.live_run import daemon_enabled,run_published


class Publisher:
    def __init__(self,name,**kw):
        self.id='fake-run';self.info={'endpoint':'http://127.0.0.1:8766'}
        self.lock=threading.Lock();self.meta={'status':'starting','step':0,'time':0.}
        self.bound=None;self.closed=None
    def bind_case(self,case):self.bound=case;return case
    def recording(self,ctx):pass
    def finish(self,*args):pass
    def update(self,**kw):self.meta.update(kw)
    def close(self,status,message=''):self.closed=(status,message)


def test_long_runs_default_to_daemon_with_explicit_opt_out():
    assert daemon_enabled(7641) and not daemon_enabled(3)
    assert daemon_enabled(3,True) and not daemon_enabled(7641,False)


def test_publication_calls_selected_runner_once_and_preserves_case(tmp_path):
    pub=Publisher('test');case=object();calls=[]
    result=NS(ctx=NS(config=NS(dt=.1)),state=NS(t=.3,state=NS(positions=np.zeros((2,2)))),
        nSteps=3,diverged=False,stopReason=None)
    def run(actual,**kwargs):calls.append((actual,kwargs));return result
    got=run_published(run,case,name='test',out=tmp_path,publisher_factory=lambda *a,**k:pub,precision='float32')
    assert got is result and calls==[(case,{'precision':'float32'})] and pub.bound is case
    assert pub.closed==('completed','')
    receipt=json.loads((tmp_path/'daemon-run.json').read_text());assert receipt['id']=='fake-run'
    assert not any(k.lower().endswith('token') for k in receipt)
    assert json.loads((tmp_path/'daemon-status.json').read_text())['status']=='completed'


@pytest.mark.parametrize('stopped',[False,True])
def test_publication_retains_original_failure_and_stop(tmp_path,stopped):
    class SimulationStopped(Exception):pass
    error=SimulationStopped('viewer stopped') if stopped else RuntimeError('numerical failure')
    pub=Publisher('test')
    def failed(*a,**kw):raise error
    with pytest.raises(type(error)) as caught:
        run_published(failed,object(),name='test',out=tmp_path,publisher_factory=lambda *a,**k:pub)
    assert caught.value is error and pub.closed[0]==('stopped' if stopped else 'failed')
