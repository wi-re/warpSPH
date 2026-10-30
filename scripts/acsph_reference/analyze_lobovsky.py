#!/usr/bin/env python3
"""Compare equal-time, equal-row physical snapshots from Lobovsky controls."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from render import read_snapshot,metrics


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--local',action='append',required=True,help='label=run_directory')
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    initial=read_snapshot(args.reference/'snapshots/000000.bin')
    reference=read_snapshot(sorted((args.reference/'snapshots').glob('*.bin'))[-1])
    f=reference['bc']==1; x0=initial['x'][f];rho_g_H=1000*9.81*.3
    data={'Reference':reference};summary={'Reference':metrics(reference)}; costs={}
    r=json.loads((args.reference/'result.json').read_text())
    costs['Reference']={'wall_seconds':r['wall_seconds'],'ms_per_step_wall':r['ms_per_step_wall'],'steps':len(r['steps']),
                        'capped_steps':sum(x['iterations']>=200 for x in r['steps'])}
    for spec in args.local:
        label,path=spec.split('=',1);path=Path(path)
        a=read_snapshot(sorted((path/'snapshots').glob('*.bin'))[-1])
        assert a['t']==reference['t'],(label,a['t'],reference['t'])
        assert np.array_equal(a['bc'],reference['bc'])
        init=read_snapshot(path/'snapshots/000000.bin')
        for key in ('x','v','p','rho','volume'):assert np.array_equal(init[key],initial[key]),(label,key)
        data[label]=a;m=metrics(a)
        for key,name in (('x','position'),('v','velocity'),('p','pressure')):
            diff=a[key][f]-reference[key][f];e2=np.sum(diff*diff,-1) if diff.ndim==2 else diff*diff
            m[name+'_rms_error']=float(np.sqrt(np.mean(e2)))
        m['pressure_rms_error_percent_rho_g_H']=100*m['pressure_rms_error']/rho_g_H
        m['position_rms_error_dx']=m['position_rms_error']/.00075
        m['ke_ratio_reference']=m['ke']/summary['Reference']['ke']
        summary[label]=m
        r=json.loads((path/'result.json').read_text());rows=r['trajectory'][1:]
        costs[label]={'wall_seconds':r['wall_seconds'],'ms_per_step_wall':1000*r['wall_seconds']/r['nSteps'],'steps':r['nSteps'],
            'capped_steps':sum(x['pseudoIterations']>=200 for x in rows),'step_ms':[x['stepTime_ms'] for x in rows],
            'graph_disabled':r.get('graph_disabled','see raw log')}
    (args.out/'summary.json').write_text(json.dumps({'metrics':summary,'costs':costs},indent=2)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(17,9),layout='constrained')
    colors=['black','#d55e00','#0072b2','#009e73','#cc79a7','#56b4e9']
    if len(data)>len(colors):
        colors=[plt.get_cmap('tab10')(i%10) for i in range(len(data))]
    for (label,a),color in zip(data.items(),colors):
        for ax,mask,coordinate,title in (
            (axes[0,0],np.isclose(x0[:,1],0),0,'Bottom material row (initial y=0)'),
            (axes[0,1],np.isclose(x0[:,0],0),1,'Left material column (initial x=0)'),
            (axes[0,2],np.isclose(x0[:,0],.3),1,'Middle material column (initial x=0.3 m)')):
            ax.plot(x0[mask,coordinate],a['p'][f][mask],label=label,color=color,lw=1.4)
            ax.set_title(title);ax.set_xlabel(('x' if coordinate==0 else 'y')+' [m]');ax.set_ylabel('Pressure [Pa]');ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=8)
    labels=list(data);where=np.arange(len(labels)); color=colors[:len(labels)]
    axes[1,0].bar(where,[summary[k]['ke'] for k in labels],color=color);axes[1,0].set_ylabel('Kinetic energy [J per unit breadth]')
    axes[1,1].bar(where,[summary[k]['vmax'] for k in labels],color=color);axes[1,1].set_ylabel('Peak speed [m/s]')
    axes[1,2].bar(where,[summary[k].get('pressure_rms_error_percent_rho_g_H',0) for k in labels],color=color);axes[1,2].set_ylabel('Pressure RMS error / rho g H [%]')
    for ax in axes[1]:ax.set_xticks(where,labels,rotation=25,ha='right',fontsize=8);ax.grid(axis='y',alpha=.25)
    fig.suptitle(f'Lobovsky startup: exact physical ICs, CUDA FP64, common t={reference["t"]:.6g} s\nAll runs hit the 200-iteration cap; this is not an impact validation',fontsize=13)
    fig.savefig(args.out/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps({'metrics':summary,'costs':costs},indent=2))

if __name__=='__main__':main()
