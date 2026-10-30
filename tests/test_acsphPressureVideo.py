"""Initial-frame ACK gating and durable pressure-video output, independent of CUDA."""
from io import BytesIO
from types import SimpleNamespace as NS
import hashlib
import json
import threading
import pytest
from scripts.acsph_reference.pressure_video import PressureVideo


def fixture(tmp_path):
    pub=NS(id='run',info={},pausing=threading.Event(),stopping=threading.Event())
    body=b'\x1aE\xdf\xa3video';manifest={'frames':[{'time':0,'step':0},{'time':.1,'step':1}]}
    job=dict(id='job',status='queued',sha256=hashlib.sha256(body).hexdigest(),artifact='/artifact',manifest='/manifest')
    state={'ack':-1};calls=[]
    def request(path,data=None,binary=False):
        calls.append((path,data))
        if binary:return BytesIO(body if path=='/artifact' else json.dumps(manifest).encode())
        if path=='/v1/runs':return {'runs':[{'id':'run','video':{'jobId':'job','ackTime':state['ack']}}]}
        return dict(job)
    video=PressureVideo(pub,tmp_path,{'scene':{'preset':'pressure'}},request=request)
    return video,pub,job,state,calls


def test_pause_precedes_job_and_only_t0_ack_resumes(tmp_path):
    video,pub,job,state,calls=fixture(tmp_path)
    assert pub.pausing.is_set() and calls[:2]==[('/v1/runs/run/pause',{}),('/v1/runs/run/video-jobs',{'scene':{'preset':'pressure'}})]
    job['status']='recording';video.tick();assert not video.resumed
    state['ack']=0;video.tick();video.tick()
    assert calls.count(('/v1/runs/run/resume',{}))==1
    assert json.loads((tmp_path/'pressure-video-start.json').read_text())['initialFrameAcked']


def test_missing_initial_frame_is_rejected(tmp_path):
    video,pub,job,state,calls=fixture(tmp_path);job['status']='recording';state['ack']=.01
    with pytest.raises(ValueError,match='initial state'):video.tick()
    assert not video.resumed


def test_completed_video_is_verified_and_retained(tmp_path):
    video,pub,job,state,calls=fixture(tmp_path);job['status']='completed';video.tick()
    assert video.done.is_set() and (tmp_path/'pressure-video/video.webm').read_bytes()==b'\x1aE\xdf\xa3video'
    assert json.loads((tmp_path/'pressure-video/manifest.json').read_text())['frames'][0]['time']==0


def test_recording_failure_stops_the_owned_run(tmp_path):
    video,pub,job,state,calls=fixture(tmp_path);job.update(status='failed',error='viewer disconnected');video.tick()
    assert video.done.is_set() and pub.stopping.is_set() and video.error=='viewer disconnected'
