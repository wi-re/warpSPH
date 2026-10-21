"""Passive pair-work export for a failed blob energy update; daemon/video on."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from compare_spheralMaterials import warp_run,save
from study_blobPartition import context,Measures,summary


def install(folder):
    from warpSPH.modules.crk import energyControl as control
    original_retain=control.retain;original_update=control.energyBeforeUpdate
    def retain(system,state,update,adjacency,index,config):
        original_retain(system,state,update,adjacency,index,config)
        stage=system.pairWork.crkEnergyStage
        stage.auditEntropy=state.pressures/state.densities.pow(5/3)
        stage.auditEnergy=state.internalEnergies
    def advance(system,update,spec,config):
        stage=system.pairWork.crkEnergyStage;s=system.state
        result=original_update(system,update,spec,config)
        if result is not None and stage.index==1 and float(result.min())<0 and not (folder/'negativeUpdate.json').exists():
            import torch
            host=lambda x:x.detach().cpu().numpy().copy()
            point=int(result.argmin());adj=stage.adjacency;i,j=adj.livePairs();mask=i==point;others=j[mask]
            half=s.velocities+.5*spec.derivative_dt*update.dvdt
            pair=stage.pairVector[:len(i)][mask]
            work=torch.einsum('ij,ij->i',half[others]-half[point],pair)
            wi=(stage.rawDudt[point]*work.sign()).clamp_min(torch.finfo(torch.float64).eps)
            wj=(stage.rawDudt[others]*work.sign()).clamp_min(torch.finfo(torch.float64).eps)
            fraction=wi/(wi+wj);increment=spec.derivative_dt*fraction*work
            entropy=stage.auditEntropy
            ef=torch.where(work>=0,entropy[others]/(entropy[point]+entropy[others]),entropy[point]/(entropy[point]+entropy[others]))
            arrays=dict(peerUIDs=host(s.UIDs[others]),peerMaterials=host(s.materials[others]),
                workSpecific=host(work),dudtFraction=host(fraction),entropyFraction=host(ef),
                increment=host(increment),peerRawDudt=host(stage.rawDudt[others]),peerEntropy=host(entropy[others]))
            np.savez_compressed(folder/'negativePairWork.npz',**arrays)
            save(folder/'negativeUpdate.json',dict(UID=int(s.UIDs[point]),material=int(s.materials[point]),
                time=float(system.t)+float(spec.derivative_dt),multiplier=float(spec.derivative_dt),
                energyBefore=float(s.internalEnergies[point]),energyAfter=float(result[point]),
                stageEnergy=float(stage.auditEnergy[point]),stageRawDudt=float(stage.rawDudt[point]),
                stageEntropy=float(entropy[point]),assignedHeating=float(increment.clamp_min(0).sum()),
                assignedCooling=float(increment.clamp_max(0).sum()),
                sameWorkEntropyIncrement=float((spec.derivative_dt*ef*work).sum()),
                limits='Counterfactual entropy weights on this one final RHS only; not a separately evolved entropy trajectory. Actual captured dudt-policy update.'))
            print('Captured negative pair-work update',folder,flush=True)
        return result
    control.retain=retain;control.energyBeforeUpdate=advance


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float32');install(a.output)
    settings=json.loads((a.output/'settings.json').read_text())
    raise SystemExit(warp_run(NS(output=a.output),settings,context,Measures,summary,'Blob partition detail'))
