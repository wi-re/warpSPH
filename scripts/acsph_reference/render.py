#!/usr/bin/env python3
"""Read full-precision dynGPUSPH snapshots; render speed/pressure video with VisPy."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import struct
import numpy as np

def read_snapshot(path):
    with path.open('rb') as f:
        n,t=struct.unpack('<id',f.read(12))
        def read(dtype,count):
            x=np.fromfile(f,dtype,count=count)
            if x.size!=count: raise ValueError(f'Truncated snapshot: {path}')
            return x
        data={'t':t,'x':read('<f8',2*n).reshape(n,2),'p':read('<f8',n),'rho':read('<f8',n),
              'v':read('<f8',2*n).reshape(n,2),'bc':read('<i4',n),'lambda':read('<f8',n),
              'volume':read('<f8',n),'aux':read('<f8',n)}
        if f.read(1): raise ValueError(f'Trailing bytes: {path}')
        return data

def metrics(d):
    f=d['bc']==1; mass=d['rho'][f]*d['volume'][f]; v=d['v'][f]; x=d['x'][f]; p=d['p'][f]
    if not all(np.isfinite(a).all() for a in (mass,v,x,p)): raise ValueError('Nonfinite snapshot')
    return {'t':d['t'],'fluid':int(f.sum()),'mass':float(mass.sum()),'volume':float(d['volume'][f].sum()),
            'ke':float(.5*np.sum(mass*np.sum(v*v,axis=1))),'vmax':float(np.linalg.norm(v,axis=1).max()),
            'pmin':float(p.min()),'pmax':float(p.max()),'pmean':float(np.average(p,weights=mass)),
            'xmin':x.min(0).tolist(),'xmax':x.max(0).tolist()}

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('run',type=Path); args=ap.parse_args()
    paths=sorted((args.run/'snapshots').glob('*.bin'))
    if not paths: raise SystemExit('No snapshots')
    stats=[metrics(read_snapshot(p)) for p in paths]
    (args.run/'snapshot-metrics.json').write_text(json.dumps(stats,indent=2)+'\n')
    from vispy import app,scene
    import matplotlib
    import imageio.v2 as imageio
    app.use_app('egl')
    canvas=scene.SceneCanvas(size=(1280,680),show=False,bgcolor='white')
    grid=canvas.central_widget.add_grid(spacing=10)
    title=scene.Label('',color='black',font_size=12); title.height_max=35; grid.add_widget(title,row=0,col=0,col_span=2)
    first=read_snapshot(paths[0]); low=first['x'].min(0); high=first['x'].max(0)
    vmax=max(s['vmax'] for s in stats) or 1
    pmax=max(max(abs(s['pmin']),abs(s['pmax'])) for s in stats) or 1
    markers=[]
    for j,label in enumerate((f'speed [0, {vmax:.4g}]',f'pressure [−{pmax:.4g}, {pmax:.4g}]')):
        text=scene.Label(label,color='black',font_size=11); text.height_max=28; grid.add_widget(text,row=1,col=j)
        view=grid.add_view(row=2,col=j,border_color='#aaa'); view.camera=scene.PanZoomCamera(aspect=1)
        view.camera.set_range(x=(low[0],high[0]),y=(low[1],high[1]),margin=.08)
        markers.append(scene.visuals.Markers(parent=view.scene,antialias=0))
    speedmap=matplotlib.colormaps['viridis']; pmap=matplotlib.colormaps['coolwarm']
    with imageio.get_writer(args.run/'output.mp4',fps=12,codec='libx264',quality=8,macro_block_size=2) as writer:
        for i,path in enumerate(paths):
            d=read_snapshot(path); f=d['bc']==1; speed=np.linalg.norm(d['v'],axis=1)
            for marker,colors in zip(markers,(speedmap(speed/vmax),pmap(.5+.5*d['p']/pmax))):
                colors[~f]=[.45,.5,.55,1]; marker.set_data(d['x'].astype('f4'),face_color=colors.astype('f4'),edge_width=0,size=2.0)
            title.text=f'{args.run.name} | dynGPUSPH 2dbc27a | t={d["t"]:.6g} | fluid={int(f.sum())}'
            frame=canvas.render(); writer.append_data(frame)
            if i in (0,len(paths)-1): imageio.imwrite(args.run/('initial.png' if i==0 else 'final.png'),frame)
    canvas.close()
    print(json.dumps({'frames':len(paths),'initial':stats[0],'final':stats[-1]}),flush=True)

if __name__=='__main__': main()
