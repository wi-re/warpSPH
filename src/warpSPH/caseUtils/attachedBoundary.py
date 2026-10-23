"""One exposed contour for a rectangular tank and a convex floor obstacle.

Construct the fluid polygon (solid outside), rather than adding the moments
of overlapping bodies. Intersections are recomputed at the actual floor;
no resolution-dependent vertical alignment of the obstacle is required.
"""
import numpy as np
import torch


def floorAttachedPolygon(lower, upper, vertices):
    lo, hi, v = np.asarray(lower, float), np.asarray(upper, float), np.asarray(vertices, float)
    floor = lo[1]
    if v[:, 1].min() > floor or v[:, 1].max() <= floor:
        return None
    if v[:, 0].min() <= lo[0] or v[:, 0].max() >= hi[0] or v[:, 1].max() >= hi[1]:
        return None  # this helper serves an obstacle attached only to the floor
    clipped = []
    for a, b in zip(v, np.roll(v, -1, axis=0)):
        if a[1] >= floor:
            clipped.append(a.copy())
        if (a[1] >= floor) != (b[1] >= floor):
            q = a+(b-a)*((floor-a[1])/(b[1]-a[1]))
            q[1] = floor
            clipped.append(q)
    # Remove duplicate cut vertices when an existing vertex lies on the floor.
    clean = []
    for q in clipped:
        if not clean or not np.allclose(q, clean[-1], rtol=0, atol=1e-12):
            clean.append(q)
    if len(clean) > 1 and np.allclose(clean[0], clean[-1], rtol=0, atol=1e-12):
        clean.pop()
    if len(clean) < 3:
        return None
    cut = [i for i, p in enumerate(clean) if abs(p[1]-floor) < 1e-12]
    if len(cut) != 2:
        return None
    left, right = sorted(cut, key=lambda i: clean[i][0])
    if clean[right][0]-clean[left][0] <= 1e-12:
        return None
    step = 1 if (left+1) % len(clean) != right else -1
    exposed, i = [clean[left]], left
    while i != right:
        i = (i+step) % len(clean)
        exposed.append(clean[i])
    return np.asarray([lo, *exposed, [hi[0], floor], hi, [lo[0], hi[1]]])


def polygonSolidSDF(vertices):
    """Exact polygon boundary distances; negative in the unbounded solid."""
    from ..geometry.sdf import getSDF, sampleSDF
    polygon = getSDF('polygon')['function']

    def sdf(x):
        v = torch.as_tensor(vertices, device=x.device, dtype=x.dtype)
        return sampleSDF(x, lambda p: -polygon(p, v))

    return sdf


def obstaclePolygonVertices(obstacle):
    """The same physical triangle used by the sampler and analytic boundary."""
    r, aspect = float(obstacle['maxExtent']), float(obstacle['aspectRatio'])
    if obstacle['obstacleType'] != 'equilateralTriangle':
        return None
    v = np.array([[-r, -r/np.sqrt(3)/aspect], [r, -r/np.sqrt(3)/aspect], [0, 2*r/np.sqrt(3)/aspect]])
    angle = -np.deg2rad(float(obstacle['aoa']))
    c, s = np.cos(angle), np.sin(angle)
    return v @ np.array([[c, -s], [s, c]]).T + [obstacle['offsetX'], obstacle['offsetY']]
