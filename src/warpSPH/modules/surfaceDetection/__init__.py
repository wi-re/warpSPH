"""Free-surface detection: color-field, color-field-gradient, Barecasco, and
Maronne schemes, plus shared normal computation, mask dilation, and the
scheme-dispatching wrapper used by the incompressible solvers.
"""

from .colorFieldDetection import detectFreeSurfaceColorField
from .colorFieldGradientDetection import detectFreeSurfaceColorFieldGradient
from .colorFieldCompute import computeColorField

from .dilation import dilateSurface
from .lambdaGrad import computeLambdaGrad, computeNormalsLambdaGrad

from .maronneNormals import computeNormalsMaronne

from .barecascoDetection import detectFreeSurfaceBarecasco
from .maronneDetection import detectFreeSurfaceMaronne

from .wrapper import detectFreeSurface
from .wp_nearestSurfaceNormal import computeNearestSurfaceNormalWarp

__all__ = [
    'detectFreeSurfaceColorField',
    'detectFreeSurfaceColorFieldGradient',
    'computeColorField',
    'dilateSurface',
    'computeLambdaGrad',
    'computeNormalsLambdaGrad',
    'computeNormalsMaronne',
    'detectFreeSurfaceBarecasco',
    'detectFreeSurfaceMaronne',
    'detectFreeSurface',
    'computeNearestSurfaceNormalWarp',
]