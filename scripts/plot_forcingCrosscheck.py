#!/usr/bin/env python3
"""Plot analytic shear accuracy and source-power sampling from a completed audit."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args();root=a.root
plt.rcParams.update({'font.size':10})
fig,axes=plt.subplots(1,3,figsize=(14,4.1),layout='constrained')
t=np.linspace(0,2,401);axes[0].plot(t,.2*np.sin(np.pi*t),'k--',label='Euler exact')
for path in sorted((root/'shear/runs').glob('*/history.jsonl')):
    h=[json.loads(s) for s in path.read_text().splitlines()];clock=[r['t'] for r in h]
    axes[0].plot(clock,[r['fittedAmplitude'] for r in h],label=path.parent.name,alpha=.8)
    axes[1].plot(clock,[r['velocityL2']/(.2/np.sqrt(2)) for r in h],label=path.parent.name)
axes[0].set(xlabel='Time',ylabel='Fitted shear amplitude',title='Independent known-answer force');axes[0].legend(fontsize=8)
axes[1].set(xlabel='Time',ylabel='Velocity L2 / exact peak RMS',title='Native shear error');axes[1].set_yscale('log');axes[1].set_ylim(bottom=1e-7);axes[1].legend(fontsize=8)
c=json.loads((root/'cadence.json').read_text());rows=[r for r in c['cadences'] if r['requestedInterval'] is not None]
axes[2].loglog([r['requestedInterval'] for r in rows],[abs(r['instantaneousSourcePower']['relativeDifference'])*100 for r in rows],'o-')
accepted=abs(c['cadences'][0]['instantaneousSourcePower']['relativeDifference'])*100
axes[2].axhline(accepted,color='k',linestyle='--',label='Every accepted state')
axes[2].set(xlabel='Requested output interval',ylabel='Absolute work difference (%)',title='Sampling versus native source work');axes[2].legend(fontsize=8)
for ax in axes:ax.grid(alpha=.2)
out=root/'figures';out.mkdir(exist_ok=True)
for suffix in ('png','pdf'):fig.savefig(out/f'forcing-crosscheck.{suffix}',dpi=170)
