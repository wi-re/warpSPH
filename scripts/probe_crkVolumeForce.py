"""Stage-only constant-pressure force audit; two tiny A/Bs, no trajectory advanced."""
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
import json
from pathlib import Path
import torch
from warpSPH.runner import RunConfiguration
from warpSPH.cases.compressible.twoGasSod import twoGasSodCase
from warpSPH.schemes.compressible.crkSPH import crkSPH_step
rows=[]
for equalMass in (False,True):
    ctx=RunConfiguration('twoGasSod',simulation={'nx':200,'tLimit':.02,'execution':'eager'},
        params={'pressureRight':1.,'equalMass':equalMass},
        solver={'crkEvolution':'spheral','crkDensityPolicy':'sameMaterial'}).prepare(twoGasSodCase)
    system=twoGasSodCase.buildSystem(ctx)
    system.initialize(ctx.config.dt,config=ctx.config,schemeConfig=ctx.schemeConfig,stageWeights=(0.,1.))
    pressureError=float((system.state.pressures-1).abs().max())
    update,_,s=crkSPH_step(system,ctx.config.dt,ctx.config,ctx.schemeConfig,stageIndex=0)
    mask=s.positions[:,0].abs()<.5
    acceleration=torch.linalg.vector_norm(update.dvdt,dim=-1)
    rows.append(dict(equalMass=equalMass,initialPressureError=pressureError,
        initialDensityRelativeMax=float((system.state.densities/ctx.scratch['prescribedDensity']-1).abs().max()),
        initialMaxAcceleration=float(acceleration[mask].max())))
p=Path('reports/materialEOS/volume/initialForce.json');p.parent.mkdir(parents=True,exist_ok=True)
p.write_text(json.dumps(dict(runs=rows),indent=2)+'\n')
print(json.dumps(rows,indent=2))
