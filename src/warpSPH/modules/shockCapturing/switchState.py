"""Per-step diagnostic/state bundle shared by the Cullen-Dehnen and Hopkins viscosity switches.

Every field besides ``alpha0s``/``alphas`` is optional because the two
switches (and the ``NoneSwitch`` no-op) populate a different subset -- e.g.
plain Cullen-Dehnen leaves ``Shear``/``Rot`` as ``None`` since it derives its
limiter from the divergence trace rather than the full shear tensor.
"""

from dataclasses import dataclass
from typing import Optional
import torch

__all__ = ['ViscositySwitchState']

@dataclass(slots = True)
class ViscositySwitchState:
    alpha0s: torch.Tensor            # Alpha0 values that get integrated over time
    alphas:  torch.Tensor            # The alpha values for the Cullen-Dehnen viscosity model
    M:       Optional[torch.Tensor]  # Gradient Correction matrix from CRKSPH
    M_inv:   Optional[torch.Tensor]  # Correction is performed using the inverse matrix
    div:     Optional[torch.Tensor]  # Divergence of the velocity field
    ddivdt:  Optional[torch.Tensor]  # Second order divergence of the velocity field
    Shear:   Optional[torch.Tensor]  # Shear tensor
    Rot:     Optional[torch.Tensor]  # Rotation tensor
    R:       Optional[torch.Tensor]  # R term from Cullen and Dehnen 2010
    Xi:      Optional[torch.Tensor]  # Limiter Term, unnamed in CRKSPH
    v_sig:   Optional[torch.Tensor]  # Signal Velocity Term
    dvdt_diss: Optional[torch.Tensor] = None  # Viscous momentum rate (ReadHayfield2012 eqs. 29-30)
    dudt_diss: Optional[torch.Tensor] = None  # Entropy-dissipation internal-energy rate (ReadHayfield2012 eqs. 33-35)


