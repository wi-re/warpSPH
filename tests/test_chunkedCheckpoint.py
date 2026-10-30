"""Real-job restart across policies, world sizes and configured offset widths."""
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import h5py
import pytest
import torch.distributed as dist


def launch(tmp_path,action,scheme,ranks,rows,offset,*,steps=2,checkpoint=None,extra=()):
    path=checkpoint or tmp_path/'checkpoint.h5'
    worker=Path(__file__).with_name('distributed')/'_checkpointWorker.py'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',
             warpSPHCore_OFFSET_DTYPE=offset)
    label=f'{action}-{ranks}-{rows}-{offset}-{path.stem}-'+sha256(repr(extra).encode()).hexdigest()[:8]
    output=tmp_path/label
    command=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc-per-node={ranks}',
        str(worker),'--action',action,'--scheme',scheme,'--rows',str(rows),'--steps',str(steps),
        '--checkpoint',str(path),'--oracle',str(tmp_path/'oracle.pt'),'--output',str(output),*extra]
    result=subprocess.run(command,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=240)
    (tmp_path/f'{label}.log').write_text(result.stdout)
    assert result.returncode==0,result.stdout
    assert result.stdout.count(f'{action} {scheme}: passed')==ranks,result.stdout
    reports=[json.loads(p.read_text()) for p in output.glob('*.json')]
    assert len(reports)==ranks and all(d['status']=='passed' for d in reports)


@pytest.mark.skipif(not dist.is_gloo_available(),reason='Gloo required')
@pytest.mark.parametrize('scheme,sourceRanks,targetRanks', [('CRKSPH',2,3),('CompSPH',1,2),('Monaghan',3,1),('GSPH',2,3),('InutsukaGSPH',3,2)])
def test_accepted_restart_changes_budget_rank_count_and_offset(tmp_path,scheme,sourceRanks,targetRanks):
    launch(tmp_path,'reference',scheme,1,4,'int64',steps=4)
    launch(tmp_path,'save',scheme,sourceRanks,3,'int32',extra=('--contracts',) if scheme=='CRKSPH' else ())
    with h5py.File(tmp_path/'checkpoint.h5','r') as file:
        assert file.attrs['complete']==1 and file.attrs['completedSteps']==2
        names=[]
        file.visit(names.append)
        assert not any(name.split('/')[-1] in ('paddedJ','paddedI','edgeOffsets','W','work') for name in names)
    launch(tmp_path,'resume',scheme,targetRanks,1,'int64',extra=('--replica-checks','contract'))
    if scheme=='CRKSPH':
        # One large chunk -> two empty compute owners, including empty W buffers.
        launch(tmp_path,'resume',scheme,3,80,'int64',extra=('--edge-budget','8192'))
        corrupt=tmp_path/'corrupt.h5'
        shutil.copy2(tmp_path/'checkpoint.h5',corrupt)
        with h5py.File(corrupt,'r+') as file:
            file['state/velocities'][0,0]+=.01
        launch(tmp_path,'reject',scheme,2,2,'int64',checkpoint=corrupt,extra=('--expect-error','checksum mismatch'))
        launch(tmp_path,'reject',scheme,2,2,'int64',extra=('--amplitude','.2','--expect-error','configuration is incompatible'))


@pytest.mark.skipif(not dist.is_gloo_available(),reason='Gloo required')
@pytest.mark.parametrize('scheme', ['CRKSPH','CompSPH','Monaghan'])
@pytest.mark.parametrize('layout,sourceOffset', [('csr','int32'),('coo','int64')])
def test_import_legacy_csr_coo_without_global_pair_read(tmp_path,scheme,layout,sourceOffset):
    extra=('--legacy',layout)
    launch(tmp_path,'reference',scheme,1,4,sourceOffset,steps=4,extra=extra)
    launch(tmp_path,'save',scheme,1,3,sourceOffset,extra=extra)
    launch(tmp_path,'resume',scheme,3,1,'int64')
    if scheme=='CRKSPH' and layout=='coo':
        for name,field,text in [('offsets','edgeOffsets','Invalid legacy CSR offsets'),
                                ('pairs','paddedJ','pair order/content'),
                                ('rows','paddedI','COO row IDs')]:
            corrupt=tmp_path/(name+'.h5')
            shutil.copy2(tmp_path/'checkpoint.h5',corrupt)
            with h5py.File(corrupt,'r+') as file:
                dataset=file['dict_sourceAdjacency/'+field]
                dataset[0]=-1
            launch(tmp_path,'reject',scheme,2,2,'int64',checkpoint=corrupt,extra=('--expect-error',text))
