"""`MeshlessConfig`, the scheme config for the meshless finite-mass / finite-volume (MFM / MFV)
scheme (`schemes/meshless.py`, COMPRESSIBLE_SPH_PLAN E1): the EOS (`gamma`), the mode (`MFM` /
`MFV`), the MUSCL order, the neighbour count `nngb` for the support re-derivation, the CFL
factor, and the pass-through parameters of the core `warpSPHCore.mfm` backend (the
`MeshlessWarp.build` closure solve and the `MeshlessWarp.rates` Riemann / limiter stage).
`MFMConfig` / `MFVConfig` pin the mode so the two registered schemes (`CompressibleSPHScheme.MFM`
/ `.MFV`) instantiate the right one with no arguments. `meshlessConfigToDict` /
`dictToMeshlessConfig` are the dict round-trip pair (with `.get` defaults) registered for both
in `schemes/builder.py`.
"""

__all__ = ['MeshlessConfig', 'MFMConfig', 'MFVConfig', 'meshlessConfigToDict', 'dictToMeshlessConfig']

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class MeshlessConfig:
    gamma: float = field(default=1.4, metadata={'description': 'Adiabatic index'})
    mode: str = field(default='MFM', metadata={"description": "'MFM' (conserved face fluxes, dm = 0) or 'MFV' (MFM fluxes + the GIZMO mass-flux limiter, dm != 0)"})
    order: int = field(default=2, metadata={"description": 'MUSCL-Hancock reconstruction order (1 = particle / Godunov, 2 = the limited gradient)'})
    nngb: float = field(default=7.0, metadata={"description": 'target neighbour count N_ngb for the support re-derivation h = (N_ngb V / C_nu)^(1/nu)'})
    cfl: float = field(default=0.2, metadata={"description": 'CFL factor for the per-particle signal timestep 2 C h / v_sig'})
    schemeName: str = field(default='Meshless SPH (MFM/MFV)', metadata={'description': 'Name of the scheme'})

    # geometry / closure solve (warpSPHCore.mfm.MeshlessWarp.build)
    closure: str = field(default='project', metadata={"description": "closure: 'project' (the graph-Laplacian CG projection) or 'none'"})
    closurePower: float = field(default=1.0, metadata={'description': 'closure power'})
    cgTol: Optional[float] = field(default=None, metadata={'description': 'closure CG tolerance (None: the backend default, 1e-6 fp64 / 1e-4 fp32)'})
    cgIters: int = field(default=200, metadata={'description': 'closure CG iteration cap'})
    condMax: float = field(default=1e3, metadata={'description': 'moment-matrix condition-number floor'})
    faceCondMax: Optional[float] = field(default=None, metadata={'description': 'per-face condition-number floor (None: the backend default)'})
    closureSolver: str = field(default='auto', metadata={"description": "closure solver: 'auto', 'cg' or 'mg'"})
    mgMinParticles: int = field(default=2048, metadata={'description': 'multigrid kick-in particle count'})
    mgMaxCoarse: int = field(default=512, metadata={'description': 'multigrid coarsest level size'})

    # Riemann / limiter stage (warpSPHCore.mfm.MeshlessWarp.rates)
    betaMin: float = field(default=1.0, metadata={'description': 'limiter beta floor'})
    betaMax: float = field(default=2.0, metadata={'description': 'limiter beta ceiling'})
    condCrit: float = field(default=100.0, metadata={'description': 'critical condition number for the limiter'})
    psi1: float = field(default=0.5, metadata={'description': 'minmod slope limiter psi_1'})
    psi2: float = field(default=0.25, metadata={'description': 'minmod slope limiter psi_2'})
    timeCentredFrame: bool = field(default=True, metadata={'description': 'evaluate the face Riemann problem in the time-centred frame'})
    retry: bool = field(default=True, metadata={'description': 'the Gaburov -> Roe -> PVRS -> Rusanov robustness retry chain'})
    massFluxLimit: float = field(default=0.1, metadata={'description': 'MFV GIZMO mass-flux limiter: max fraction of a donor mass moved per step (0 = off)'})


@dataclass
class MFMConfig(MeshlessConfig):
    # the conserved face-flux scheme: mass is conserved (dm = 0)
    mode: str = field(default='MFM', metadata={"description": "pinned to 'MFM'"})


@dataclass
class MFVConfig(MeshlessConfig):
    # the finite-volume scheme: MFM fluxes + the GIZMO mass-flux limiter (dm != 0)
    mode: str = field(default='MFV', metadata={"description": "pinned to 'MFV'"})


def meshlessConfigToDict(config: MeshlessConfig) -> Dict[str, Any]:
    return {
        'gamma': config.gamma,
        'mode': config.mode,
        'order': config.order,
        'nngb': config.nngb,
        'cfl': config.cfl,
        'schemeName': config.schemeName,
        'closure': config.closure,
        'closurePower': config.closurePower,
        'cgTol': config.cgTol,
        'cgIters': config.cgIters,
        'condMax': config.condMax,
        'faceCondMax': config.faceCondMax,
        'closureSolver': config.closureSolver,
        'mgMinParticles': config.mgMinParticles,
        'mgMaxCoarse': config.mgMaxCoarse,
        'betaMin': config.betaMin,
        'betaMax': config.betaMax,
        'condCrit': config.condCrit,
        'psi1': config.psi1,
        'psi2': config.psi2,
        'timeCentredFrame': config.timeCentredFrame,
        'retry': config.retry,
        'massFluxLimit': config.massFluxLimit,
    }


def dictToMeshlessConfig(configDict: Dict[str, Any]) -> MeshlessConfig:
    config = MeshlessConfig()
    config.gamma = configDict.get('gamma', config.gamma)
    config.mode = configDict.get('mode', config.mode)
    config.order = configDict.get('order', config.order)
    config.nngb = configDict.get('nngb', config.nngb)
    config.cfl = configDict.get('cfl', config.cfl)
    config.schemeName = configDict.get('schemeName', config.schemeName)
    config.closure = configDict.get('closure', config.closure)
    config.closurePower = configDict.get('closurePower', config.closurePower)
    config.cgTol = configDict.get('cgTol', config.cgTol)
    config.cgIters = configDict.get('cgIters', config.cgIters)
    config.condMax = configDict.get('condMax', config.condMax)
    config.faceCondMax = configDict.get('faceCondMax', config.faceCondMax)
    config.closureSolver = configDict.get('closureSolver', config.closureSolver)
    config.mgMinParticles = configDict.get('mgMinParticles', config.mgMinParticles)
    config.mgMaxCoarse = configDict.get('mgMaxCoarse', config.mgMaxCoarse)
    config.betaMin = configDict.get('betaMin', config.betaMin)
    config.betaMax = configDict.get('betaMax', config.betaMax)
    config.condCrit = configDict.get('condCrit', config.condCrit)
    config.psi1 = configDict.get('psi1', config.psi1)
    config.psi2 = configDict.get('psi2', config.psi2)
    config.timeCentredFrame = configDict.get('timeCentredFrame', config.timeCentredFrame)
    config.retry = configDict.get('retry', config.retry)
    config.massFluxLimit = configDict.get('massFluxLimit', config.massFluxLimit)
    return config
