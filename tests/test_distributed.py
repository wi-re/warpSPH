"""Small separate-process Gloo solver A/Bs; no GPU or NCCL overlay required."""
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch.distributed as dist


@pytest.mark.skipif(not dist.is_gloo_available(),reason='Gloo required')
@pytest.mark.parametrize('steps', [2, 3])
@pytest.mark.parametrize('checks',['exact','contract'])
def test_replicated_eager_solver_ownership_and_replay(tmp_path, steps,checks):
    worker=Path(__file__).with_name('distributed')/'_solverWorker.py'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=2',
         str(worker),'--replica-checks',checks,'--device','cpu','--output',str(tmp_path),'--steps',str(steps)]
    completed=subprocess.run(cmd,env=env,text=True,stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,timeout=240)
    assert completed.returncode==0,completed.stdout
    assert completed.stdout.count('all solver checks passed')==2,completed.stdout


@pytest.mark.skipif(not dist.is_gloo_available(),reason='Gloo required')
@pytest.mark.parametrize('ranks',[2,3])
@pytest.mark.parametrize('checks',['exact','contract'])
def test_filtered_owner_redistribution_replays_fresh_stages(tmp_path,ranks,checks):
    worker=Path(__file__).with_name('distributed')/'_solverWorker.py'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    command=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc-per-node={ranks}',
        str(worker),'--replica-checks',checks,'--device','cpu','--output',str(tmp_path),
        '--steps','2','--filter-replay','--clock-interval','.00025']
    result=subprocess.run(command,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=240)
    (tmp_path/'run.log').write_text(result.stdout)
    assert result.returncode==0,result.stdout
    assert result.stdout.count('all solver checks passed')==ranks,result.stdout


def _launch_document(tmp_path,name,document,ranks,*,rows=3,edgeBudget=1024,expectSuccess=True,output=None):
    import json
    configuration=tmp_path/(name+'.json')
    configuration.write_text(json.dumps(document))
    output=output or tmp_path/name
    script=Path(__file__).resolve().parents[1]/'scripts/run_distributed.py'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    command=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc-per-node={ranks}',
        str(script),'--configuration',str(configuration),'--backend','gloo','--output',str(output),
        '--edge-budget',str(edgeBudget),'--max-rows',str(rows),'--checkpoint-every','1','--timeout-seconds','45',
        '--replica-checks','contract' if name=='first' else 'exact']
    result=subprocess.run(command,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=240)
    (tmp_path/(name+'.log')).write_text(result.stdout)
    assert (result.returncode==0)==expectSuccess,result.stdout
    return output


@pytest.mark.skipif(not dist.is_gloo_available(),reason='Gloo required')
@pytest.mark.parametrize('scheme',['CRKSPH','CompSPH','Monaghan'])
def test_launcher_outputs_and_restart_are_rank_owned(tmp_path,scheme):
    import json
    import h5py
    import torch
    document=dict(schemaVersion=1,case='drivenTurbulence',simulation=dict(nx=8,dim=2,nSteps=4,
        maxDt=.0002,plot=False,video=False,store=False,execution='eager',cudaGraph=False,
        progress=True,stallProgress=1e-3,velocityAlarmPlotInterval=1,integrationScheme='rungeKutta4'),
        params=dict(accelerationRMS=.1,forcingInterval=.00025),solver=dict(scheme=scheme))
    reference=_launch_document(tmp_path,'reference',document,1,rows=4)
    document['simulation']['nSteps']=2
    first=_launch_document(tmp_path,'first',document,2)
    document['simulation']['resumeFrom']=str(first/'finalState.h5')
    resumed=_launch_document(tmp_path,'resumed',document,3,rows=1)
    for path,ranks,completed,rows in [(reference,1,4,5),(first,2,2,3),(resumed,3,4,3)]:
        manifest=json.loads((path/'manifest.json').read_text())
        assert manifest['status']=='complete' and manifest['completed']==completed
        assert len(manifest['placement'])==ranks
        assert len({r['pid'] for r in manifest['placement']})==ranks
        assert all(r['backend']=='gloo' for r in manifest['placement'])
        assert all(p['digest'] and p['files'] for p in manifest['sources'].values())
        reports=[json.loads(p.read_text()) for p in path.glob('rank-*.json')]
        assert len(reports)==ranks and all(r['status']=='complete' for r in reports)
        assert len((path/'diagnostics.jsonl').read_text().splitlines())==rows
        assert len(list(path.glob('checkpoint-*.h5')))==completed-(2 if path==resumed else 0)
    with h5py.File(reference/'finalState.h5') as a,h5py.File(resumed/'finalState.h5') as b:
        assert a.attrs['time']==b.attrs['time']
        for name in a['state']:
            tol=2e-12 if name=='sourceEnergyBudget' else 0
            torch.testing.assert_close(torch.as_tensor(a['state'][name][()]),torch.as_tensor(b['state'][name][()]),rtol=tol,atol=tol)
        for name in ('forcing/knots','forcing/generator','clock/time','clock/nextDt'):
            torch.testing.assert_close(torch.as_tensor(a[name][()]),torch.as_tensor(b[name][()]),rtol=0,atol=0)
    if scheme=='CRKSPH':
        before=(first/'manifest.json').read_bytes()
        _launch_document(tmp_path,'existing-output',document,2,expectSuccess=False,output=first)
        assert (first/'manifest.json').read_bytes()==before
        document['simulation'].pop('resumeFrom')
        failed=_launch_document(tmp_path,'failed-capacity',document,2,edgeBudget=1,expectSuccess=False)
        assert json.loads((failed/'manifest.json').read_text())['status']=='failed'
        assert len(list(failed.glob('failure-*.json')))==2 and not (failed/'finalState.h5').exists()


@pytest.mark.skipif(not dist.is_gloo_available(),reason='Gloo required')
@pytest.mark.parametrize('checks',['exact','contract'])
def test_native_rhs_allocation_failure_propagates_and_fresh_retry_recovers(tmp_path,checks):
    worker=Path(__file__).with_name('distributed')/'_solverWorker.py'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=3',str(worker),
        '--rhs-fault','--device','cpu','--replica-checks',checks,'--output',str(tmp_path),'--clock-interval','.00025']
    result=subprocess.run(command,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=240)
    (tmp_path/'run.log').write_text(result.stdout)
    assert result.returncode==0,result.stdout
    assert result.stdout.count('all solver checks passed')==3,result.stdout
