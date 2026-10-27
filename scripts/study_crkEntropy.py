"""Recorded 1D reset/raw-heating/compatible entropy ledgers; no model changes."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from compare_spheralMaterials import save,warp_run
from compare_spheralSod1d import exact
from study_crkEnergySuite import summarize as base_summary
from study_owenDimensions import context as dimension_context,SOURCES as DIMENSION_SOURCES
from study_idealH import study
from crkEntropyLedger import install

SOURCES=tuple(dict.fromkeys((__file__,'scripts/crkEntropyLedger.py','scripts/study_owenDimensions.py',*DIMENSION_SOURCES)))

def hashes():
    return {str(Path(p)):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in SOURCES}


def prepare(root):
    root.mkdir(parents=True,exist_ok=False);rows=[]
    reference=Path('results/materialEOS/owen_dimensions_2026-10-21/sod-1d-hermiteShell').resolve()
    old=json.loads((reference/'settings.json').read_text())
    for case in ('sod','contact'):
        if case=='sod':initial=dict(np.load(reference/'input.npz'))
        else:
            x=(-.5+(np.arange(100)+.5)/100)[:,None];tag=(abs(x[:,0])<.25).astype(np.int32)
            rho=np.where(tag,4.,1.);P=np.ones(len(x));n=len(x)
            initial=dict(positions=x,velocities=np.zeros_like(x),densities=rho,pressures=P,
                internalEnergies=P/(.4*rho),supports=np.full(n,.04),masses=rho*.01,
                materials=tag,UIDs=np.arange(n,dtype=np.int64),kinds=np.zeros(n,dtype=np.int32))
        for split in (1,2):
            name=f'{case}-dt{split}';folder=root/name;folder.mkdir();np.savez_compressed(folder/'input.npz',**initial)
            steps=200*split;lo,hi=(-1.5,1.5) if case=='sod' else (-.5,.5)
            s=dict(old,case=case,domain=[lo,hi],domainAxes=[[lo,hi]],end=.2,steps=steps,exactSteps=steps,
                dt=.2/steps,timeGrid=[.2/steps]*steps,frameEvery=5*split,policy='entropyLedger',
                gammaVariant=f'1D {case} entropy dt/{split}',inputPath=str((folder/'input.npz').resolve()),
                comparisonPolicy='Read-only accepted-path entropy ledger; generated material-aware Owen, paper density, '
                    'fresh midpoint, geometric volumes, native compatible energy and averaged-Q thermal predictor fixed')
            save(folder/'settings.json',s);rows.append(name)
    (root/'sources').mkdir();source=hashes()
    for name,h in source.items():
        p=Path(name);shutil.copyfile(p,root/'sources'/f'{p.stem}_{h[:12]}{p.suffix}')
    save(root/'provenance.json',dict(rows=rows,sources=source,baseline=str(reference),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        scope='Observational hooks only. Half-step predictor is a separate branch; entropy telescopes through '
            'reset, raw full-step density/pressure/AV heating and final compatible correction. '
            'No entropy repair, rescaling of u or numerical default changes. Two timestep levels, one spatial resolution.'))


def context(s):
    from warpSPH.runner import RunConfiguration
    case,_,sim,params,solver=dimension_context(s)
    case=replace(case,name='crkEntropy_'+s['case'])
    return case,RunConfiguration(case.name,simulation=sim,params=params,solver=solver).prepare(case),sim,params,solver


class Measures:
    def __init__(self,s,initial):
        self.s=s;self.initial=initial;self.m=initial['masses'];self.mass=self.m.sum()
        self.E=np.dot(self.m,initial['internalEnergies']+.5*np.sum(initial['velocities']**2,axis=1))
        self.p=np.sum(self.m[:,None]*initial['velocities'],axis=0)
        self.s0=np.log(initial['pressures']/initial['densities']**s['gamma'])
    def __call__(self,a,t):
        m,rho,u,v,x=(a[k] for k in ('masses','densities','internalEnergies','velocities','positions'))
        P=a['pressures'];ds=np.log(P/rho**self.s['gamma'])-self.s0;V=m/rho
        row=dict(time=float(t),relativeEnergyDrift=float(np.dot(m,u+.5*np.sum(v*v,axis=1))/self.E-1),
            momentumError=float(np.linalg.norm(np.sum(m[:,None]*v,axis=0)-self.p)/self.mass),
            materialMassError=float(abs(m.sum()/self.mass-1)),minDensity=float(rho.min()),minEnergy=float(u.min()),
            minPressure=float(P.min()),maxVelocity=float(np.linalg.norm(v,axis=1).max()),
            entropyChangePerMass=float(np.dot(m,ds)/self.mass),entropyL1PerMass=float(np.dot(m,abs(ds))/self.mass),
            negativeEntropyFraction=float(m[ds<-1.e-8].sum()/self.mass),minSupport=float(a['supports'].min()),
            maxSupport=float(a['supports'].max()))
        if self.s['case']=='sod':
            q,_=exact(x[:,0],t);mask=abs(x[:,0])<.5
            for k,values in [('density',rho),('pressure',P),('energy',u),('velocity',v[:,0])]:
                row[k+'L1']=float(np.dot(V[mask],abs(values[mask]-q[k][mask]))/V[mask].sum())
        else:
            L=self.s['domain'][1]-self.s['domain'][0];delta=(x-self.initial['positions']+L/2)%L-L/2
            row.update(pressureError=float(abs(P-1).max()),densityRelativeL1=float(np.mean(abs(rho/self.initial['densities']-1))),
                displacementInSpacings=float(abs(delta).max()*self.s['nx']))
        return row


def run_row(args):
    sys.path.insert(0,str(args.publisher_path));s=json.loads((args.output/'settings.json').read_text())
    assert hashes()==json.loads((args.output.parent/'provenance.json').read_text())['sources']
    ledger=install(s)
    def initialize(c,system,a,settings):
        system.state.crkVolumes=None
        c.scratch['_titleProvenance']='Entropy accounting | '+s['gammaVariant']+' | fresh midpoint / native compatible'
    def summary(root,solver,settings,history,initial,final,wall,steps,remote,extra):
        row=base_summary(root,solver,settings,history,initial,final,wall,steps,remote,extra)
        l=ledger.summary();expected=dict(resets=steps,rhs=2*steps,updates=2*steps,finalizations=steps,negativeEOSCalls=0,supportCalls=2*steps)
        row['entropyLedger']=l;row['gates'].update(entropyClosure=l['maximumCumulativeClosure']<1.e-12 and l['maximumEndpointResidual']<1.e-12,
            counts=l['counters']==expected,energy=row['maxAbsoluteEnergyDrift']<1.e-10,
            identity=all(np.array_equal(initial[k],final[k]) for k in ('masses','materials','UIDs')))
        if s['case']=='contact':row['gates']['stationary']=max(h['maxVelocity'] for h in history)<1.e-9
        row['passed']=all(row['gates'].values());save(root/'comparisonRun.json',row)
        print('ENTROPY RESULT',s['case'],s['gammaVariant'],'passed',row['passed'],'budget',l['groups']['globalBox'],flush=True)
        return row
    try:return warp_run(args,s,context,Measures,summary,'CRK entropy accounting',initialize_factory=initialize)
    finally:ledger.save(args.output)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','study','row','verify'))
    p.add_argument('--output',required=True,type=Path);p.add_argument('--report',type=Path,default=Path('reports/materialEOS/entropyAccounting'))
    p.add_argument('--publisher-path',type=Path,default=Path('/home/lu26029/dev/warpSPHRemote/python'))
    a=p.parse_args();a.output=a.output.resolve()
    if a.mode=='prepare':prepare(a.output)
    elif a.mode=='study':study(a.output,row_script=__file__)
    elif a.mode=='row':raise SystemExit(run_row(a))
    else:
        sys.path.insert(0,str(a.publisher_path));from analyze_crkEnergySuite import verify
        verify(a.output,a.report.resolve())


if __name__=='__main__':main()
