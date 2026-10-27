"""CPU-only figures/summary for static 1D CRK operator qualification."""
import argparse
import json
from pathlib import Path
import numpy as np


def analyze(root,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True,exist_ok=True)
    summaries=[]
    fig,axes=plt.subplots(2,2,figsize=(10,6.5),constrained_layout=True)
    fields=('constantInterpolation','linearInterpolation','constantGradientNaive','linearGradientDifference')
    targets=('1','0.7+1.3x','0','1.3')
    for source in sorted(root.glob('*/operators.json')):
        result=json.loads(source.read_text())
        rows=[r for r in result['records'] if r['variant']=='crk']
        assert len(rows)==24 and all(r['passed'] for r in rows)
        summary=dict(precision=result['precision'],device=result['device'],coherentCases=len(rows),
            worstErrors={key:max(r['regions'][region][key]['max'] for r in rows for region in ('interface','bulk'))
                         for key in rows[0]['regions']['interface']},
            source=str(source.resolve()),ratio4={r['volume']:r for r in rows
                if r['spacingRatio']==4 and r['momentSupport']=='Scatter'},
            mismatchRatio4=[r for r in result['records'] if r['variant']=='mismatchedGather' and r['spacingRatio']==4])
        summaries.append(summary)
        selected=summary['ratio4']['physical']
        a=np.load(selected['data']);mask=a['observed']
        for ax,key,target in zip(axes.flat,fields,targets):
            expected=1. if key=='constantInterpolation' else .7+1.3*a['x'] if key=='linearInterpolation' else 0. if key.startswith('constant') else 1.3
            error=np.abs(a[key]-expected)
            ax.semilogy(a['x'][mask],np.maximum(error[mask],1.e-18),'.-',ms=2,lw=.6,
                        label=result['precision']+' / '+result['device'])
            ax.set(title=key+'; target '+target,xlabel='x',ylabel='absolute error')
        interface=2*a['supports'].max()
    for ax in axes.flat:
        ax.axvspan(-interface,interface,color='grey',alpha=.12)
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Production CRK operators: 4:1 spacing jump, physical volumes, scatter\nShading: interface region; errors below 1e-18 displayed at 1e-18')
    fig.savefig(output/'operatorErrors.png',dpi=160);plt.close(fig)
    (output/'summary.json').write_text(json.dumps(dict(runs=summaries),indent=2)+'\n')
    print(json.dumps([{k:r[k] for k in ('precision','device','coherentCases','worstErrors')} for r in summaries],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();analyze(a.root,a.output)
