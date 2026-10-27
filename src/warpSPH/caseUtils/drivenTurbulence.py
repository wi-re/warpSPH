"""Case-local SPH state adapters; Q is an output view, never an evolved field."""
from dataclasses import dataclass
from typing import Optional
import torch
from warpSPHIntegrators.state import constant
from ..systems.compSPH import CompSPHState
from ..systems.compressibleMonaghan import CompressibleState


class ConservedOutput:
    @property
    def Q(self):
        m, v, u = self.masses, self.velocities, self.internalEnergies
        return torch.cat((m[:, None], m[:, None]*v,
                          (m*(u+.5*v.square().sum(-1)))[:, None]), dim=1)


@dataclass
class DrivenCompSPHState(ConservedOutput, CompSPHState):
    sourceEnergyBudget: Optional[torch.Tensor] = constant(tags=('source_energy_budget',), default=None)


@dataclass
class DrivenGasState(ConservedOutput, CompressibleState):
    sourceEnergyBudget: Optional[torch.Tensor] = constant(tags=('source_energy_budget',), default=None)


def adjacencyCheckpoint(system):
    a = system.adjacency
    data = dict(checkpointAdjacencyVersion=1, checkpointAdjacencyPresent=a is not None)
    if a is not None:
        names = ('paddedI', 'paddedJ', 'numNeighbors', 'edgeOffsets', 'queryPositions',
                 'referencePositions', 'querySupports', 'referenceSupports')
        data['sourceAdjacency'] = {name: getattr(a, name).detach().cpu() for name in names}
        data['sourceAdjacency'].update(numRows=a.numRows, numCols=a.numCols,
            numEdges=a.numEdges if a.numEdges is not None else a.paddedJ.numel(),
            samePositions=a.queryPositions is a.referencePositions,
            sameSupports=a.querySupports is a.referenceSupports)
    return data


def restoreAdjacency(system, checkpoint):
    from warpSPHCore.dataTypes import AdjacencyList
    if checkpoint.attrs.get('checkpointAdjacencyVersion') != 1:
        raise ValueError('drivenTurbulence SPH checkpoint is missing native neighbor history')
    if not checkpoint.attrs['checkpointAdjacencyPresent']:
        system.adjacency = None
        return
    if 'dict_sourceAdjacency' not in checkpoint:
        raise ValueError('drivenTurbulence SPH checkpoint has incomplete native neighbor history')
    saved = checkpoint['dict_sourceAdjacency']
    args = {name: torch.from_numpy(value[:]).to(system.state.positions.device) for name,value in saved.items()}
    for name in ('numRows','numCols','numEdges'):
        args[name] = int(saved.attrs[name])
    n = len(system.state.positions)
    if (args['numRows'] != n or args['numCols'] != n or args['numEdges'] < 0
            or args['numEdges'] > args['paddedJ'].numel()):
        raise ValueError('drivenTurbulence SPH checkpoint neighbor dimensions are incompatible')
    if saved.attrs['samePositions']:
        args['referencePositions'] = args['queryPositions']
    if saved.attrs['sameSupports']:
        args['referenceSupports'] = args['querySupports']
    system.adjacency = AdjacencyList(**args)
