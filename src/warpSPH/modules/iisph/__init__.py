"""Reusable native IISPH solver, static wall coupling and predictor terms.

Numerical policy is explicit in IISPHConfig, passed to each solve. Spatial
state belongs to an IISPHOperator instance; stages own no global run state.
"""

from .solver import IISPHOperator, IISPHResult
from .boundary import IISPHBoundary, buildIISPHBoundary
from .viscosity import iisphViscosity

__all__ = ['IISPHOperator', 'IISPHResult', 'IISPHBoundary',
           'buildIISPHBoundary', 'iisphViscosity']
