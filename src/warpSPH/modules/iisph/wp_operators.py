"""Device kernels for IISPH's rest-volume operators (no pressure matrix).

Neighbor discovery, kernel functions, scalar precision and device ownership
come from warpSPHCore. These reductions specialize the IISPH coefficients;
they do not introduce a second neighbor search or kernel implementation.
"""

from functools import lru_cache
import warp as wp
from warpSPHCore.type_config import scalar_t
from warpSPHCore.kernels.kernel import sphKernel_
from warpSPHCore.kernels.gradient import sphGradient_


@lru_cache(None)
def kernels(dim):
    V = wp.types.vector(length=dim, dtype=scalar_t)
    I = wp.types.vector(length=dim, dtype=wp.int32)

    @wp.func
    def displacement(a: V, b: V, length: V, periodic: I):
        d = a - b
        for k in range(wp.static(dim)):
            if periodic[k] != 0:
                d[k] -= wp.round(d[k] / length[k]) * length[k]
        return d

    @wp.kernel
    def geometry(x: wp.array(dtype=V), reference: wp.array(dtype=V),
                 h: wp.array(dtype=scalar_t), ii: wp.array(dtype=wp.int64), jj: wp.array(dtype=wp.int64),
                 length: V, periodic: I, kernel: wp.int32, norm: scalar_t,
                 grad: wp.array(dtype=V), weights: wp.array(dtype=scalar_t)):
        e = wp.tid()
        i = ii[e]
        d = displacement(x[i], reference[jj[e]], length, periodic)
        grad[e] = norm * sphGradient_(d, h[i], kernel)
        weights[e] = norm * sphKernel_(d, h[i], kernel)

    @wp.kernel
    def coefficients(jj: wp.array(dtype=wp.int64), starts: wp.array(dtype=wp.int64),
                     counts: wp.array(dtype=wp.int64), grad: wp.array(dtype=V),
                     volume: wp.array(dtype=scalar_t),
                     kinds: wp.array(dtype=wp.int32), sampled: wp.bool,
                     G: wp.array(dtype=V),
                     density: wp.array(dtype=scalar_t), wallG: wp.array(dtype=V),
                     c: wp.array(dtype=V), diagonal: wp.array(dtype=scalar_t)):
        i = wp.int64(wp.tid())
        s = density[i]
        boundary = G[i]
        total = V(scalar_t(0.0))
        squares = scalar_t(0.0)
        for k in range(wp.int32(counts[i])):
            e = starts[i] + wp.int64(k)
            j = jj[e]
            if kinds[j] == 0:
                total += volume[j] * grad[e]
                squares += volume[j] * wp.dot(grad[e], grad[e])
            elif sampled and kinds[j] == 1:
                boundary += volume[j] * grad[e]
        total += boundary
        wallG[i] = boundary
        c[i] = total
        diagonal[i] = scalar_t(0.0)
        if kinds[i] == 0:
            diagonal[i] = -(wp.dot(total, total) + volume[i] * squares) / (s * s)

    @wp.kernel
    def rate(jj: wp.array(dtype=wp.int64), starts: wp.array(dtype=wp.int64),
             counts: wp.array(dtype=wp.int64), grad: wp.array(dtype=V),
             volume: wp.array(dtype=scalar_t), kinds: wp.array(dtype=wp.int32),
             G: wp.array(dtype=V), field: wp.array(dtype=V), output: wp.array(dtype=scalar_t)):
        i = wp.int64(wp.tid())
        value = scalar_t(0.0)
        if kinds[i] == 0:
            value = wp.dot(field[i], G[i])
            for k in range(wp.int32(counts[i])):
                e = starts[i] + wp.int64(k)
                j = jj[e]
                if kinds[j] == 0:
                    value += volume[j] * wp.dot(field[i] - field[j], grad[e])
        output[i] = value

    @wp.kernel
    def force(jj: wp.array(dtype=wp.int64), starts: wp.array(dtype=wp.int64),
              counts: wp.array(dtype=wp.int64), grad: wp.array(dtype=V),
              volume: wp.array(dtype=scalar_t), kinds: wp.array(dtype=wp.int32),
              s: wp.array(dtype=scalar_t), c: wp.array(dtype=V), G: wp.array(dtype=V),
              contact: wp.array(dtype=scalar_t), q: wp.array(dtype=scalar_t), output: wp.array(dtype=V)):
        i = wp.int64(wp.tid())
        value = V(scalar_t(0.0))
        if kinds[i] == 0:
            value = -c[i] * q[i] / (s[i] * s[i]) - G[i] * contact[i]
            for k in range(wp.int32(counts[i])):
                e = starts[i] + wp.int64(k)
                j = jj[e]
                if kinds[j] == 0:
                    value -= volume[j] * q[j] / (s[j] * s[j]) * grad[e]
        output[i] = value

    @wp.kernel
    def contact_denominator(jj: wp.array(dtype=wp.int64), starts: wp.array(dtype=wp.int64),
                            counts: wp.array(dtype=wp.int64), weights: wp.array(dtype=scalar_t),
                            volume: wp.array(dtype=scalar_t), kinds: wp.array(dtype=wp.int32),
                            G: wp.array(dtype=V), denominator: wp.array(dtype=scalar_t)):
        i = wp.int64(wp.tid())
        value = scalar_t(0.0)
        if kinds[i] == 0 and wp.dot(G[i], G[i]) > scalar_t(1.e-24):
            for k in range(wp.int32(counts[i])):
                e = starts[i] + wp.int64(k)
                j = jj[e]
                if kinds[j] == 0:
                    value += volume[j] * weights[e]
        denominator[i] = value

    @wp.kernel
    def contact_value(jj: wp.array(dtype=wp.int64), starts: wp.array(dtype=wp.int64),
                      counts: wp.array(dtype=wp.int64), weights: wp.array(dtype=scalar_t),
                      volume: wp.array(dtype=scalar_t), kinds: wp.array(dtype=wp.int32),
                      denominator: wp.array(dtype=scalar_t), q: wp.array(dtype=scalar_t),
                      output: wp.array(dtype=scalar_t)):
        i = wp.int64(wp.tid())
        value = scalar_t(0.0)
        if denominator[i] > scalar_t(0.0):
            for k in range(wp.int32(counts[i])):
                e = starts[i] + wp.int64(k)
                j = jj[e]
                if kinds[j] == 0:
                    value += volume[j] * weights[e] * q[j]
            value /= denominator[i]
        output[i] = value

    @wp.kernel
    def contact_diagonal(jj: wp.array(dtype=wp.int64), starts: wp.array(dtype=wp.int64),
                         counts: wp.array(dtype=wp.int64), grad: wp.array(dtype=V),
                         volume: wp.array(dtype=scalar_t), kinds: wp.array(dtype=wp.int32),
                         x: wp.array(dtype=V), contact: wp.array(dtype=V), h: wp.array(dtype=scalar_t),
                         G: wp.array(dtype=V), c: wp.array(dtype=V), denominator: wp.array(dtype=scalar_t),
                         length: V, periodic: I, kernel: wp.int32, norm: scalar_t,
                         diagonal: wp.array(dtype=scalar_t)):
        i = wp.int64(wp.tid())
        delta = scalar_t(0.0)
        if kinds[i] == 0:
            if denominator[i] > scalar_t(0.0):
                d = displacement(x[i], contact[i], length, periodic)
                tii = volume[i] * norm * sphKernel_(d, h[i], kernel) / denominator[i]
                delta -= tii * wp.dot(c[i], G[i])
            for k in range(wp.int32(counts[i])):
                e = starts[i] + wp.int64(k)
                j = jj[e]
                if kinds[j] == 0 and denominator[j] > scalar_t(0.0):
                    d = displacement(x[i], contact[j], length, periodic)
                    tji = volume[i] * norm * sphKernel_(d, h[j], kernel) / denominator[j]
                    delta += volume[j] * tji * wp.dot(grad[e], G[j])
        diagonal[i] += delta

    @wp.kernel
    def sampled_gravity(jj: wp.array(dtype=wp.int64), starts: wp.array(dtype=wp.int64),
                        counts: wp.array(dtype=wp.int64), grad: wp.array(dtype=V),
                        volume: wp.array(dtype=scalar_t), kinds: wp.array(dtype=wp.int32),
                        x: wp.array(dtype=V), contact: wp.array(dtype=V), gravity: wp.array(dtype=V),
                        length: V, periodic: I,
                        output: wp.array(dtype=V)):
        i = wp.int64(wp.tid())
        value = V(scalar_t(0.0))
        if kinds[i] == 0:
            for k in range(wp.int32(counts[i])):
                e = starts[i] + wp.int64(k)
                j = jj[e]
                if kinds[j] == 1:
                    d = displacement(x[j], contact[i], length, periodic)
                    value -= volume[j] * wp.dot(gravity[i], d) * grad[e]
        output[i] = value

    return dict(V=V, I=I, geometry=geometry, coefficients=coefficients, rate=rate, force=force,
                contact_denominator=contact_denominator, contact_value=contact_value,
                contact_diagonal=contact_diagonal, sampled_gravity=sampled_gravity)
