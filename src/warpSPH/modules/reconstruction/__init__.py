"""Pair-velocity policies for the pairwise dissipation operator (AV_PLAN Phase 1 / 3): which velocity
difference `u_ij` the artificial-viscosity term sees. `rawPairVelocity` is the historical behaviour;
the limited-linear reconstruction (Frontiere 2017, Garcia-Senz & Cabezon 2026 Eqs. 11-19) lands here
in Phase 3, extracted from `modules/crk`."""

from .pairVelocity import rawPairVelocity

__all__ = ['rawPairVelocity']
