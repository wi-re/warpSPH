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
    cfl: float = field(default=0.1, metadata={"description": "CFL factor C of the per-particle signal timestep dt = 2 C L / v_sig (GIZMO's CourantFac: 0.1 in its KH / Gresho tests, 0.2 its documented maximum recommended value); L is set by `cflLength`"})
    cflLength: str = field(default='particle', metadata={"description": "length L of the CFL timestep: 'particle' (default; GIZMO's Get_Particle_Size, L = V^(1/d) from the moment volume -- ~4x smaller than the kernel radius at ~50 neighbours) or 'support' (the kernel radius h of the core driver's `2 C h / v_sig`; what the core-parity scripts compare against, and ~4x GIZMO's step at the same C -- its Hancock predictor shows stripe noise at C = 0.2)"})
    schemeName: str = field(default='Meshless SPH (MFM/MFV)', metadata={'description': 'Name of the scheme'})

    # temporal integration (D4: the step owns the whole time step, so the
    # integrator must be single-evaluation -- see `schemes/meshless.py`). The
    # per-mode default comes from the D4 measurement (COMPRESSIBLE_SPH_PLAN).
    temporalScheme: str = field(default='auto', metadata={"description": "the temporal scheme the step performs: 'auto' (per-mode default from the D4 measurement: MFM -> 'ssprk', MFV -> 'hancock'), 'hancock' (the core driver's single-stage MUSCL-Hancock + Euler; the G1 port-fidelity setting), 'ssprk' (two-stage SSPRK, rates evaluated at dt = 0), or 'kdk' (kick-drift-kick, rates at dt = 0)"})

    # geometry / closure solve (warpSPHCore.mfm.MeshlessWarp.build)
    closure: str = field(default='none', metadata={"description": "face closure: 'none' (default; Hopkins 2015 / GIZMO -- the effective faces as the moment matrices give them, no projection; matches GIZMO on the Yee vortex and KH 256, COMPRESSIBLE_SPH_PLAN 2026-10-12) or 'project' (the core's graph-Laplacian projection enforcing sum_j A_ij = 0 exactly: not in GIZMO, lower velocity error on near-lattice flow but flips / amplifies faces on disordered roll-ups -- opt-in)"})
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

    # EXPERIMENTAL switches of the E2 wave-2 D confirmation runs (default off = the shipped behaviour)
    faceGuard: int = field(default=0, metadata={'description': "EXPERIMENTAL: re-guard the closure-corrected face A' = A_g + |A_g| (lam_j - lam_i): 0 off, 1 drop the correction on pairs with A'.d <= 0, 2 scale it to keep A'.d >= 0.25 A_g.d, 3 / 4 cap |A' - A_g| at 0.5 / 0.25 |A_g|"})
    closureCap: float = field(default=0.0, metadata={'description': "EXPERIMENTAL bounded closure: 0 = the shipped unbounded projection; c > 0 enters the face correction through the vector clip of the potential difference to magnitude c (A' = A_g + kappa clip(lam_j - lam_i, c), solved by lagged secant weights), so |A' - A_g| <= c |A_g| (power 1): no reversed / over-amplified face while the cell closure is kept wherever the bound is not saturated"})
    mfvUpdate: str = field(default='conserved', metadata={'description': "EXPERIMENTAL (MFV, hancock): 'conserved' integrates Q with the capped mass flux and the full momentum / energy; 'primitive' is GIZMO's kick: v += dt (F_p - v F_m,full) / m_new, u += dt (F_E - v.F_p + v^2/2 F_m,full - u F_m,full) / m (u halved at most), mass capped per pair and floored at -0.99 m"})


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
        'cflLength': config.cflLength,
        'schemeName': config.schemeName,
        'temporalScheme': config.temporalScheme,
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
        'faceGuard': config.faceGuard,
        'closureCap': config.closureCap,
        'mfvUpdate': config.mfvUpdate,
    }


def dictToMeshlessConfig(configDict: Dict[str, Any]) -> MeshlessConfig:
    config = MeshlessConfig()
    config.gamma = configDict.get('gamma', config.gamma)
    config.mode = configDict.get('mode', config.mode)
    config.order = configDict.get('order', config.order)
    config.nngb = configDict.get('nngb', config.nngb)
    config.cfl = configDict.get('cfl', config.cfl)
    config.cflLength = configDict.get('cflLength', config.cflLength)
    config.schemeName = configDict.get('schemeName', config.schemeName)
    config.temporalScheme = configDict.get('temporalScheme', config.temporalScheme)
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
    config.faceGuard = configDict.get('faceGuard', config.faceGuard)
    config.closureCap = configDict.get('closureCap', config.closureCap)
    config.mfvUpdate = configDict.get('mfvUpdate', config.mfvUpdate)
    return config
