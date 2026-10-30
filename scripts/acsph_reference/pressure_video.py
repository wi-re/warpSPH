"""Request/retain a viewer video without changing the numerical runner.

An initialized run remains paused until the viewer has encoded and acknowledged
its exact t=0 frame. No browser lease is acquired by this controller.
"""
import hashlib
import json
from pathlib import Path
import threading
from urllib.request import Request,urlopen


class PressureVideo:
    def __init__(self,publisher,out,settings,request=None):
        self.publisher=publisher;self.out=Path(out);self.done=threading.Event()
        self.error=None;self.resumed=False;self.result=None
        self.request=request or self._request
        self.base=f'/v1/runs/{publisher.id}'
        self.request(self.base+'/pause',{})
        publisher.pausing.set()
        self.job=self.request(self.base+'/video-jobs',settings)
        self.path=self.base+'/video-jobs/'+self.job['id']
        self.receipt(self.job)

    def _request(self,path,data=None,binary=False):
        info=self.publisher.info
        req=Request(info['endpoint']+path,data=None if data is None else json.dumps(data).encode(),
            headers={'Authorization':'Bearer '+info['token'],'Content-Type':'application/json'})
        response=urlopen(req,timeout=10)
        if binary:return response
        with response:return json.load(response)

    def receipt(self,job):
        tmp=self.out/'pressure-video-job.pending'
        tmp.write_text(json.dumps(job,indent=2,allow_nan=False)+'\n');tmp.replace(self.out/'pressure-video-job.json')

    def tick(self):
        if self.done.is_set():return
        job=self.request(self.path);self.receipt(job)
        if job['status'] in ('failed','cancelled'):
            self.error=job.get('error','Pressure video '+job['status']);self.done.set();self.publisher.stopping.set();return
        if job['status']=='recording' and not self.resumed:
            runs=self.request('/v1/runs')['runs'];run=next(r for r in runs if r['id']==self.publisher.id)
            video=run.get('video') or {}
            if video.get('jobId')==job['id'] and video.get('ackTime',-1)>=0:
                if video['ackTime']!=0:raise ValueError('Pressure recording missed its initial state.')
                self.request(self.base+'/resume',{});self.resumed=True
                (self.out/'pressure-video-start.json').write_text(json.dumps(dict(initialFrameAcked=True,time=0,job=job['id']))+'\n')
        if job['status']=='completed':
            folder=self.out/'pressure-video';folder.mkdir(exist_ok=True)
            pending=folder/'video.webm.pending';digest=hashlib.sha256()
            with self.request(job['artifact'],binary=True) as source,pending.open('wb') as target:
                for chunk in iter(lambda:source.read(65536),b''):target.write(chunk);digest.update(chunk)
            if digest.hexdigest()!=job['sha256']:raise ValueError('Pressure video checksum mismatch.')
            with self.request(job['manifest'],binary=True) as source:manifest=json.load(source)
            if not manifest['frames'] or manifest['frames'][0]['time']!=0:raise ValueError('Pressure video does not begin at t=0.')
            pending.replace(folder/'video.webm')
            (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            (folder/'job.json').write_text(json.dumps(job,indent=2)+'\n')
            self.result=job;self.done.set()
