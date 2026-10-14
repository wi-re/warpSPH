"""Iterative smoothing-length (h) solvers that couple h to local density via a target neighbor count.

Dispatches between the Owen (lookup-table based) and Monaghan (Newton-iteration
on the density-support constraint) schemes on
``compParams.adaptiveSupportScheme``.
"""

from .optimalSupport import evaluateOptimalSupport
from .stage import solveStageSupport
from .wp_omega import computeOmegaWarp as computeOmega

__all__ = ['evaluateOptimalSupport', 'solveStageSupport', 'computeOmega']