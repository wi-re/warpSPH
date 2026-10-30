"""Small separate-process Gloo solver A/Bs; no GPU or NCCL overlay required."""
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch.distributed as dist


@pytest.mark.skipif(not dist.is_gloo_available(),reason='Gloo required')
@pytest.mark.parametrize('steps', [2, 3])
def test_replicated_eager_solver_ownership_and_replay(tmp_path, steps):
    worker=Path(__file__).with_name('distributed')/'_solverWorker.py'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=2',
         str(worker),'--device','cpu','--output',str(tmp_path),'--steps',str(steps)]
    completed=subprocess.run(cmd,env=env,text=True,stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,timeout=240)
    assert completed.returncode==0,completed.stdout
    assert completed.stdout.count('all solver checks passed')==2,completed.stdout
