"""Fresh reconstruction and flux evaluation for a meshless temporal stage."""
import torch
from warpSPHCore.profiling import record_function
from warpSPHCore.type_config import scalar_t, get_torch_precision
from ...utils.syncFree import deviceConstant
from ..reconstruction.meshless import _LIMITER_SPEC
from .flux import fluxSpec


def deviceTimestep(dt, device):
    if isinstance(dt, torch.Tensor):
        if dt.numel() != 1:
            raise ValueError('meshless timestep must be scalar')
        return dt.to(device=device, dtype=get_torch_precision()).reshape(1)
    return deviceConstant([dt], dtype=get_torch_precision(), device=device)


def computeRates(geometry, rho: torch.Tensor, vel: torch.Tensor, pres: torch.Tensor, gamma: float, dt: float = 0.0,
          mode: str = "MFM", order: int = 2, beta_min: float = 1.0, beta_max: float = 2.0,
          cond_crit: float = 100.0, psi1: float = 0.5, psi2: float = 0.25, timeCentredFrame: bool = True,
          retry: bool = True, massFluxLimit: float = 0.1, faceGuard: int = 0, materials=None):
    """``dQ/dt`` of shape ``(N, 2 + dim)`` for ``Q = (m, m v, m e_tot)``, as ``scheme.mfmRates``."""
    geometry._checkForwardOnly(rho, vel, pres, dt if isinstance(dt, torch.Tensor) else None,
                               gamma if isinstance(gamma, torch.Tensor) else None)
    mode = mode.upper()
    if mode not in ("MFM", "MFV"):
        raise ValueError("mode must be 'MFM' or 'MFV'")
    N, dim = vel.shape
    gammas = gamma.to(rho).contiguous() if isinstance(gamma, torch.Tensor) else torch.full_like(rho, gamma)
    if gammas.shape != rho.shape:
        raise ValueError('meshless gamma field must match the particle density shape')
    if materials is None:
        materials = torch.zeros(N, dtype=torch.int32, device=vel.device)
    rho, vel, pres = rho.contiguous(), vel.contiguous(), pres.contiguous()
    with record_function('[warpSPH] - [meshless rates - 01] reconstruction'):
        if order >= 2:
            gRho, gVel, gP = geometry.gradients(rho, vel, pres)
            aRho, aVel, aP = geometry._fast.launch(_LIMITER_SPEC, rho=rho, vel=vel, pres=pres, gRho=gRho, gVel=gVel,
                                               gP=gP, cond=geometry.cond, betaMin=scalar_t(beta_min),
                                               betaMax=scalar_t(beta_max), condCrit=scalar_t(cond_crit), rowStart=0)
        else:
            gRho = torch.zeros(N, dim, dtype=rho.dtype, device=rho.device)
            gVel = torch.zeros(N, dim, dim, dtype=rho.dtype, device=rho.device)
            gP = torch.zeros_like(gRho)
            aRho, aP = torch.ones_like(rho), torch.ones_like(rho)
            aVel = torch.ones_like(vel)
    with record_function('[warpSPH] - [meshless rates - 02] Riemann flux'):
        dm, dp, de = geometry._fast.launch(
            fluxSpec(geometry._fast.useAdjacency), rho=rho, vel=vel, pres=pres, gRho=gRho, gVel=gVel, gP=gP, aRho=aRho, aVel=aVel, aP=aP,
            vol=geometry.volume, Einv=geometry.Einv, lam=geometry.lam, condBad=geometry.condBad, gamma=gammas, materials=materials, dt=deviceTimestep(dt, rho.device),
            mode=int(0 if mode.upper() == "MFM" else 1), order=int(order), psi1=scalar_t(psi1), psi2=scalar_t(psi2),
            power=scalar_t(geometry.closure_power), timeCentred=int(timeCentredFrame), centred=int(geometry.centred),
            areaCap=int(geometry.areaCap), retry=int(retry), massLimit=scalar_t(massFluxLimit),
            faceGuard=int(faceGuard), closureCap=scalar_t(geometry.closure_cap), rowStart=0)
    return torch.cat([dm[:, None], dp, de[:, None]], dim=-1)

