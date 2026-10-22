"""Explicit policies for distinct pressure iterations.

Defaults describe the existing local variants, not claims of paper fidelity.
Legacy scheme-level experiment switches are read by compatibility wrappers
at each call; numerical modules never import a scheme to obtain policy.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class OmniPressurePolicy:
    OMEGA: float = 0.3
    RESIDUAL_FLOOR: float = -1e-3
    ALPHA_FLOOR: float = 1e-25
    VERBOSE_SOLVE: bool = False
    WALL_PRESSURE_MODE: str | None = "shepard"
    CD_SOURCE_PROJECT: str = "auto"
    CD_PROJECT_THRESHOLD: float = 0.7
    CD_SOLVER: str = "jacobi"
    CD_KRYLOV_MAXITER: int = 256
    CD_KRYLOV_RTOL: float = 1e-3
    CD_KRYLOV_RESTART: int = 50
    CD_TIKHONOV: float = 0.0


@dataclass(frozen=True)
class BandPressurePolicy:
    OMEGA_FLUID: float = 0.1
    DENSITY_MIN_ITERATIONS: int = 8
    DENSITY_MAX_ITERATIONS: int = 256
    ALPHA_FLOOR: float = 1e-25
    RESIDUAL_FLOOR: float = -1e-3
    DIAG_TIKHONOV: float = 0.05
    CLOSED_DOMAIN_GAUGE: str = "auto"
    NULL_MODE_THRESHOLD: float = 0.25
