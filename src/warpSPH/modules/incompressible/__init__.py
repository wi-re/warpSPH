"""IISPH-style implicit pressure solvers: a divergence-free variant and a
constant-density (density-invariance) variant, both driven by a relaxed
Jacobi iteration over the predicted-velocity divergence/density error.
"""

from .divergenceFree import solveDivergenceFree
from .incompressible import solveIncompressible
# Compatibility exports; the native implementation belongs to modules.iisph.
from ..iisph import IISPHOperator, IISPHResult, IISPHBoundary, buildIISPHBoundary

__all__ = ['solveDivergenceFree', 'solveIncompressible', 'IISPHOperator', 'IISPHResult',
           'IISPHBoundary', 'buildIISPHBoundary']
