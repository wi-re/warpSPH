"""Static qualification and standalone cost of the NumPy ring closure solve."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from crkClosure1D import close_edges,pressure_rhs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for path in sorted(args.input.glob('n*-ratio*-*.npz')):
        a=dict(np.load(path));x=a['x'];V=a['volume'];i=a['i'];j=a['j'];q0=a['publishedCRK_edge']
        times=[]
        for _ in range(11):
            start=time.perf_counter();q=close_edges(x,V,i,j,q0,2.);times.append(time.perf_counter()-start)
        rho=np.where(x<0,1.,.25);mass=V*rho;mask=abs(x)<.5
        ac,_,_=pressure_rhs(V,mass,np.ones_like(x),np.zeros_like(x),i,j,q)
        al,_,_=pressure_rhs(V,mass,1+x,np.zeros_like(x),i,j,q)
        P=1+.1*np.sin(np.pi*x);v=.2*np.sin(2*np.pi*x)
        ap,dV,du=pressure_rhs(V,mass,P,v,i,j,q);exact=-.1*np.pi*np.cos(np.pi*x)/rho
        row=dict(case=path.stem,particles=len(x),edges=len(i),medianSolveMilliseconds=1000*float(np.median(times[1:])),
            constantAccelerationMax=float(abs(ac[mask]).max()),affineAccelerationErrorMax=float(abs(al[mask]+1/rho[mask]).max()),
            sineAccelerationL1=float(np.sum(V*abs(ap-exact))/V.sum()),
            relativeEdgeChangeL2=float(np.linalg.norm(q-q0)/np.linalg.norm(q0)),
            totalForce=float(np.dot(mass,ap)),totalVolumeRate=float(dV.sum()),
            totalPower=float(np.dot(mass*v,ap)+np.dot(mass,du)))
        assert max(row['constantAccelerationMax'],row['affineAccelerationErrorMax'])<1.e-8,row
        assert max(abs(row[k]) for k in ('totalForce','totalVolumeRate','totalPower'))<1.e-10,row
        np.savez_compressed(args.output/path.name,**a,structured_edge=q,
                            structured_constantAcceleration=ac,structured_affineAcceleration=al,structured_sineAcceleration=ap)
        rows.append(row);print(json.dumps(row),flush=True)
    benchmark=[]
    for n in (256,1024,4096,16384,65536):
        nr=n//4;N=n+nr
        x=np.r_[-1+(np.arange(n)+.5)/n,(np.arange(nr)+.5)/nr]
        V=np.r_[np.full(n,1/n),np.full(nr,1/nr)]
        i=np.tile(np.arange(N),3);j=np.concatenate([(np.arange(N)+k)%N for k in (1,2,3)])
        # Synthetic regular-stencil coefficients for solve cost only, not a
        # new physical CRK baseline on the irregular geometry.
        q0=np.repeat([.15625,.125,.03125],N)
        times=[]
        for _ in range(6):
            start=time.perf_counter();close_edges(x,V,i,j,q0,2.);times.append(time.perf_counter()-start)
        row=dict(particles=N,edges=len(i),medianSolveMilliseconds=1000*float(np.median(times[1:])))
        benchmark.append(row);print('cost '+json.dumps(row),flush=True)
    (args.output/'structuredClosure.json').write_text(json.dumps(dict(cases=rows,costOnlyBenchmark=benchmark,
        limits='Static geometry and pressure-work checks. Benchmark uses synthetic starting edge coefficients. NumPy sort/map plus linear prefix solve; not a GPU runtime comparison.'),indent=2)+'\n')


if __name__=='__main__':main()
