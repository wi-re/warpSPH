"""Render the retained static density derivative arrays, without recomputing physics."""
import argparse
from pathlib import Path
import numpy as np


def plot(report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=report/'operators'
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for row,t in enumerate(('.06','.2')):
        name='sod-dt1-compatible-t'+str(float(t));a=dict(np.load(folder/(name+'.npz')));x=a['positions'][:,0];order=np.argsort(x);order=order[(x[order]>-.3)&(x[order]<.4)]
        axes[row,0].plot(x[order],a['warp_paperDensity'][order],'.-',label='Geometric corrected density')
        axes[row,0].plot(x[order],a['native_stockDensity'][order],'x',label='Native stock same-material density')
        axes[row,0].set(title=f'Saved midpoint near t={t}: density estimators',ylabel='Density',xlabel='x')
        ax=axes[row,1]
        for key,label in [('warp_gradient','Continuity: -velocity gradient'),('warp_fixed_logDensityRate','Geometric derivative: R fixed'),('warp_measuredSupport_logDensityRate','Geometric derivative: measured R rate')]:
            value=-a[key] if key=='warp_gradient' else a[key]
            ax.plot(x[order],value[order],'.-',ms=3,label=label)
        ax.plot(x[order],a['native_fixed_logDensityRate'][order],'x',ms=3,label='Native components: R fixed')
        ax.set(title=f'Material log-density rates near t={t}',ylabel='1 / time',xlabel='x')
        for ax in axes[row]:ax.legend(fontsize=8)
    fig.savefig(report/'densityRates.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,name in zip(axes,('uniformAffine','unequalSpacingAffine')):
        a=dict(np.load(folder/(name+'.npz')));x=a['positions'][:,0];mask=a['mask']
        ax.plot(x[mask],-a['warp_gradient'][mask],'-',label='Continuity (exact affine value)')
        ax.plot(x[mask],a['warp_fixed_logDensityRate'][mask],'.',label='Geometric derivative: R fixed')
        ax.plot(x[mask],a['warp_measuredSupport_logDensityRate'][mask],'x',label='Geometric derivative: R scales with x')
        ax.set(title='Uniform spacing' if name=='uniformAffine' else '4:1 spacing jump, equal mass',xlabel='x',ylabel='Material log-density rate')
        ax.set_ylim(-.14,-.06)
        ax.ticklabel_format(axis='y',style='plain',useOffset=False)
        ax.legend(fontsize=8)
    fig.savefig(report/'affineControls.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,default=Path('reports/materialEOS/densityEnergy'));a=p.parse_args();plot(a.report)
