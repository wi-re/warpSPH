"""IISPH-style implicit pressure solvers: a divergence-free variant and a
constant-density (density-invariance) variant, both driven by a relaxed
Jacobi iteration over the predicted-velocity divergence/density error.
"""

from .divergenceFree import solveDivergenceFree
from .incompressible import solveIncompressible
__all__ = ['solveDivergenceFree', 'solveIncompressible']
