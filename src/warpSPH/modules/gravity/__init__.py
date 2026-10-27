"""Body-force gravity terms (directional, point-source, and radial
potential-field), selected and dispatched by `wrapper.computeGravity` based
on `schemeConfig.gravityConfig`.
"""

from .wrapper import computeGravity
from .bodyForce import computeBodyForce, bodyForceVector

__all__ = ['computeGravity', 'computeBodyForce', 'bodyForceVector']