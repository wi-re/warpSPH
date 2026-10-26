"""Independent native DFSPH density/divergence policies and warm starts."""

from dataclasses import dataclass, field
import math


@dataclass
class DFSPHProjectionConfig:
    minIterations: int = 1
    maxIterations: int = 100
    maxErrorPercent: float = 0.1
    # At n_h=4, the native periodic operator's measured Jacobi limits are
    # .353 (C2) / .468 (C4). The reference .5 is outside both windows.
    relaxationFactor: float = 0.3
    clampPressure: bool = True

    def validate(self):
        if not (isinstance(self.minIterations, int) and isinstance(self.maxIterations, int)
                and 1 <= self.minIterations <= self.maxIterations):
            raise ValueError('DFSPH requires 1 <= minIterations <= maxIterations')
        if not math.isfinite(self.maxErrorPercent) or self.maxErrorPercent < 0:
            raise ValueError('DFSPH maxErrorPercent must be finite and nonnegative')
        if not math.isfinite(self.relaxationFactor) or not 0 < self.relaxationFactor <= 1:
            raise ValueError('DFSPH relaxationFactor must be in (0, 1]')


@dataclass
class DFSPHConfig:
    density: DFSPHProjectionConfig = field(default_factory=lambda: DFSPHProjectionConfig(
        minIterations=2, maxErrorPercent=.01))
    divergence: DFSPHProjectionConfig = field(default_factory=DFSPHProjectionConfig)
    deficiencyGuard: bool = True
    warmStart: bool = True
    # Reference cutoff on |S|² + Vi sum_j Vj |grad W_ij|², in 1/length².
    # Applied before density-ratio or timestep scaling; zero restores the
    # historical exact-zero-only policy. This is not a pressure ceiling.
    weakFactorThreshold: float = 1.e-5
    # Sampling-recovery rule shared by every native DFSPH composition.
    # Density includes sampled/analytic wall completion. Applies only to
    # clamped divergence; use zero for the original ungated reference.
    # Below this rho/rho0, pin the pressure unknown (including warm starts)
    # to zero. Neighbour pressure forces and density projection are unchanged.
    # Signed projection ignores it; density projection is unchanged.
    minDivergenceDensityRatio: float = 0.9
    # Alternate compositions are opt-in; the native/published cycle stays default.
    ordering: str = 'df_np_cd'

    def validate(self):
        if not math.isfinite(self.minDivergenceDensityRatio) or not 0 <= self.minDivergenceDensityRatio <= 1:
            raise ValueError('DFSPH minDivergenceDensityRatio must be in [0, 1]')
        if not math.isfinite(self.weakFactorThreshold) or self.weakFactorThreshold < 0:
            raise ValueError('DFSPH weakFactorThreshold must be finite and nonnegative')
        if self.ordering not in ('df_np_cd', 'np_df_cd', 'np_df_shift'):
            raise ValueError('Unknown DFSPH ordering')
        self.density.validate()
        self.divergence.validate()
        if self.density.minIterations < 2:
            raise ValueError('DFSPH density projection requires at least 2 iterations')


def dictToDFSPHConfig(values):
    config = DFSPHConfig(**{key: value for key, value in values.items()
                           if key not in ('density', 'divergence')})
    for name in ('density', 'divergence'):
        if name in values:
            defaults = vars(getattr(config, name))
            setattr(config, name, DFSPHProjectionConfig(**{**defaults, **values[name]}))
    config.validate()
    return config
