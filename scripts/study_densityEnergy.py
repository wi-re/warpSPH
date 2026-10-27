"""Preferred-support 1D energy matrix and density/continuity operator captures."""
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
from study_crkEntropy import Measures,context as old_context,SOURCES as ENTROPY_SOURCES
from study_crkEnergySuite import POLICIES,summarize as base_summary
from study_idealH import study
from crkEnergyDensityAudit import install

SOURCES=tuple(dict.fromkeys((__file__,'scripts/crkEnergyDensityAudit.py','scripts/study_crkEntropy.py',
    'scripts/crkThermalRate.py','src/warpSPH/modules/compSPH/pairWork.py',*ENTROPY_SOURCES)))

def hashes():return {str(Path(p)):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in SOURCES}


def prepare(root):
    root.mkdir(parents=True,exist_ok=False);rows=[]
    parent=Path('results/materialEOS/entropy_accounting_2026-10-21').resolve()
    for case in ('sod','contact'):
        for split in (1,2):
            source=parent/f'{case}-dt{split}';old=json.loads((source/'settings.json').read_text())
            for policy,choice in POLICIES.items():
                name=f'{case}-dt{split}-{policy}';folder=root/name;folder.mkdir();shutil.copyfile(source/'input.npz',folder/'input.npz')
                s=dict(old,policy=policy,choice=choice,gammaVariant=f'1D {case} {policy} dt/{split}',
                    inputPath=str((folder/'input.npz').resolve()),comparisonPolicy='Only thermal derivative/final compatible correction varies; '
                    'preferred generated material-aware before-RHS Owen, paper density, geometric volumes, fresh midpoint, AV fixed')
                save(folder/'settings.json',s);rows.append(name)
    (root/'sources').mkdir();source=hashes()
    for name,h in source.items():
        p=Path(name);shutil.copyfile(p,root/'sources'/f'{p.stem}_{h[:12]}{p.suffix}')
    save(root/'provenance.json',dict(rows=rows,sources=source,parent=str(parent),policies=POLICIES,
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        scope='Four-way thermal derivative/compatible-update matrix at two dt grids on two 1D cases. '
            'entropy is an existing research continuous work-distribution ODE, not paper Eq78 or unchanged Spheral. '
            'Shadow compatible reconstruction on every raw/compatible trajectory is read-only. No numerical defaults change.'))


def context(s):
    from warpSPH.runner import RunConfiguration
    case,_,sim,params,solver=old_context(s)
    assert solver['compatibleEnergy']==s['choice']['compatible']
    case=replace(case,name='densityEnergy_'+s['case'])
    return case,RunConfiguration(case.name,simulation=sim,params=params,solver=solver).prepare(case),sim,params,solver


def run_row(args):
    sys.path.insert(0,str(args.publisher_path));s=json.loads((args.output/'settings.json').read_text())
    assert hashes()==json.loads((args.output.parent/'provenance.json').read_text())['sources']
    ledger=install(s,args.output)
    def initialize(c,system,a,settings):
        system.state.crkVolumes=None
        c.scratch['_titleProvenance']='Density / energy | '+s['gammaVariant']+' | fresh midpoint / generated Owen'
        save(args.output/'resolvedPolicy.json',dict(compatible=c.schemeConfig.compatibleEnergy,thermal=s['choice']['thermal'],
            partition=c.schemeConfig.crkEnergyPartition,predictor=c.schemeConfig.crkEnergyPredictor,
            density=c.schemeConfig.crkEvolution,table=c.schemeConfig.owenTable,materialWeights=c.schemeConfig.owenMaterialWeights,
            integration=c.config.integrationScheme.name))
    def summarize(root,solver,settings,history,initial,final,wall,steps,remote,extra):
        row=base_summary(root,solver,settings,history,initial,final,wall,steps,remote,extra);l=ledger.summary()
        expected=dict(resets=steps,rhs=2*steps,updates=2*steps,finalizations=steps,negativeEOSCalls=0,supportCalls=2*steps)
        row['entropyLedger']=l;row['shadowCompatibleMaximumClosure']=max((abs(v['shadowEnergyRelativeClosure']) for v in ledger.shadowRows),default=0.)
        row['gates'].update(entropyClosure=l.get('maximumCumulativeClosure',1.e100)<1.e-12 and l.get('maximumEndpointResidual',1.e100)<1.e-12,
            counts=l['counters']==expected,shadowEnergy=bool(ledger.shadowRows) and row['shadowCompatibleMaximumClosure']<1.e-10,
            identity=all(np.array_equal(initial[k],final[k]) for k in ('masses','materials','UIDs')))
        if s['choice']['compatible']:row['gates']['energy']=row['maxAbsoluteEnergyDrift']<1.e-10
        if s['case']=='contact':row['gates']['stationary']=max(h['maxVelocity'] for h in history)<1.e-9
        row['passed']=all(row['gates'].values());save(root/'comparisonRun.json',row)
        print('DENSITY ENERGY RESULT',s['gammaVariant'],'passed',row['passed'],'E',row['maxAbsoluteEnergyDrift'],flush=True)
        return row
    try:return warp_run(args,s,context,Measures,summarize,'CRK density / energy',initialize_factory=initialize)
    finally:
        ledger.save(args.output);save(args.output/'shadowEnergy.json',ledger.shadowRows);save(args.output/'frozenStates.json',ledger.frozen)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','study','row','verify'))
    p.add_argument('--output',required=True,type=Path);p.add_argument('--report',type=Path,default=Path('reports/materialEOS/densityEnergy'))
    p.add_argument('--publisher-path',type=Path,default=Path('/home/lu26029/dev/warpSPHRemote/python'))
    a=p.parse_args();a.output=a.output.resolve()
    if a.mode=='prepare':prepare(a.output)
    elif a.mode=='study':study(a.output,row_script=__file__)
    elif a.mode=='row':raise SystemExit(run_row(a))
    else:
        sys.path.insert(0,str(a.publisher_path));from analyze_crkEnergySuite import verify
        verify(a.output,a.report.resolve())


if __name__=='__main__':main()
