"""NumPy continuum references for future MHD controls; no production solver API.

All vectors have three components, independently of spatial dimension. Magnetic
permeability is explicit: E_B=|B|^2/(2*mu). No cleaning energy is assumed.
Sources: Hopkins & Raives (2016), arXiv:1505.02783; Athena++ cpaw.cpp;
PLUTO Linear_Force_Free reference. See docs/mhd_scaffolding.md for scope.
"""
import math
import numpy as np


def _scalar(value, name, *, zero=False):
    if not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f'{name} must be finite and {"nonnegative" if zero else "positive"}')
    return value


def _vector(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim == 0 or value.shape[-1] != 3 or not np.isfinite(value).all():
        raise ValueError('Require finite vectors with three components')
    return value


def magnetic_energy_density(field, *, mu):
    return np.sum(_vector(field)**2, axis=-1)/(2*_scalar(mu, 'mu'))


def magnetic_increment_energy(field, increment, *, mu):
    """Exact finite energy increment at fixed volume, including the quadratic term."""
    b, db = _vector(field), _vector(increment)
    return np.sum((b+.5*db)*db, axis=-1)/_scalar(mu, 'mu')


def wave_speeds(*, density, pressure, gamma, field, normal, mu):
    """Fluid-frame (slow, normal Alfven, fast) speeds for a unit normal."""
    rho = _scalar(density, 'density')
    p = _scalar(pressure, 'pressure', zero=True)
    _scalar(gamma-1, 'gamma-1')
    mu = _scalar(mu, 'mu')
    b, n = _vector(field), _vector(normal)
    if not np.allclose(np.sum(n*n, axis=-1), 1., rtol=1e-13, atol=1e-13):
        raise ValueError('normal must have unit length')
    bn = np.sum(b*n, axis=-1)
    a2 = gamma*p/rho
    b2 = np.sum(b*b, axis=-1)/(mu*rho)
    bn2 = bn*bn/(mu*rho)
    bt2 = np.sum((b-bn[...,None]*n)**2, axis=-1)/(mu*rho)
    # This form avoids cancellation for parallel, nearly degenerate waves.
    discriminant = (a2-b2)**2 + 4*a2*bt2
    fast2 = .5*(a2+b2+np.sqrt(discriminant))
    slow2 = np.divide(a2*bn2, fast2, out=np.zeros_like(fast2), where=fast2>0)
    return np.sqrt(slow2), np.sqrt(bn2), np.sqrt(fast2)


def circular_alfven(x, t, *, length, density, pressure, parallel_field,
                    amplitude, bulk_velocity, direction, mu):
    """Exact finite-amplitude x-directed ideal-MHD wave, uniform rho and P.

    The returned velocity and field are (N,3), even for a 1D coordinate array.
    The wave travels at bulk_velocity+direction*parallel_field/sqrt(mu*rho).
    """
    _scalar(length, 'length'); _scalar(density, 'density'); _scalar(mu, 'mu')
    _scalar(pressure, 'pressure'); _scalar(parallel_field, 'parallel_field')
    if direction not in (-1, 1) or not all(np.isfinite([t, amplitude, bulk_velocity])):
        raise ValueError('Require direction +/-1 and finite wave parameters')
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError('x must be a finite 1D array')
    scale = math.sqrt(mu*density)
    phase = 2*math.pi/length*(x-(bulk_velocity+direction*parallel_field/scale)*t)
    b = np.column_stack((np.full_like(x, parallel_field), amplitude*np.cos(phase), amplitude*np.sin(phase)))
    v = -direction*b/scale
    v[:,0] = bulk_velocity
    return dict(density=np.full_like(x,density), pressure=np.full_like(x,pressure), velocity=v, field=b)


def potential_mode(points, t, *, lengths, amplitude, drift=(0.,0.)):
    """Curl of periodic A_z=A*sin(kx*(x-vx*t))*sin(ky*(y-vy*t)).

    Also usable as a divergence-free induction-source shape, with A in units
    of vector potential per time. This is a prescribed induction reference,
    not a generic equilibrium of the fully coupled momentum equations.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('Require finite (N,2) coordinates')
    if len(lengths) != 2 or len(drift) != 2:
        raise ValueError('Require two lengths and two drift components')
    for length in lengths: _scalar(length, 'length')
    if not np.isfinite([t,amplitude,*drift]).all():
        raise ValueError('Require finite mode parameters')
    kx, ky = 2*math.pi/np.asarray(lengths)
    x, y = (points-np.asarray(drift)*t).T
    return np.column_stack((amplitude*ky*np.sin(kx*x)*np.cos(ky*y),
                           -amplitude*kx*np.cos(kx*x)*np.sin(ky*y), np.zeros(len(x))))


def resistive_mode(x, t, *, length, density, internal_energy, amplitude,
                   magnetic_diffusivity, mu):
    """Exact periodic force-free decay B=(0,a*cos(k*x),a*sin(k*x)).

    eta_m is magnetic diffusivity [L^2/time], not electrical resistivity.
    Uniform Joule heating preserves total energy with v=0 and uniform pressure.
    No cooling, cleaning, physical conduction or viscosity is included.
    """
    _scalar(length,'length'); _scalar(density,'density'); _scalar(mu,'mu')
    _scalar(internal_energy,'internal_energy')
    _scalar(magnetic_diffusivity,'magnetic_diffusivity',zero=True)
    _scalar(t,'time',zero=True)
    x = np.asarray(x,dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all() or not math.isfinite(amplitude):
        raise ValueError('Require finite x array and amplitude')
    k = 2*math.pi/length
    exponent = -magnetic_diffusivity*k*k*t
    a = amplitude*math.exp(exponent)
    b = np.column_stack((np.zeros_like(x),a*np.cos(k*x),a*np.sin(k*x)))
    heat = -amplitude**2*math.expm1(2*exponent)/(2*mu*density)
    return dict(field=b, internal_energy=np.full_like(x,internal_energy+heat),
                joule_heating_rate=magnetic_diffusivity*k*k*a*a/(mu*density))


def divergence_summary(divergence, *, length_scale, volume, reference_field, estimator):
    """Summarize an externally supplied divergence; do not invent its operator.

    A declared nonzero B_ref keeps the normalization defined in zero-field
    controls. Signed integrated divergence cannot replace local error norms.
    """
    div, h, vol = np.broadcast_arrays(np.asarray(divergence,dtype=np.float64),
                                     np.asarray(length_scale,dtype=np.float64),
                                     np.asarray(volume,dtype=np.float64))
    _scalar(reference_field,'reference_field')
    if (div.size == 0 or not np.isfinite([div,h,vol]).all()
            or np.any(h<=0) or np.any(vol<=0) or not isinstance(estimator,str) or not estimator.strip()):
        raise ValueError('Require finite divergence, positive lengths/volumes and named estimator')
    error = h*div/reference_field
    return dict(estimator=estimator, reference_field=reference_field,
                dimensional_rms=float(np.sqrt(np.sum(vol*div*div)/vol.sum())),
                normalized_rms=float(np.sqrt(np.sum(vol*error*error)/vol.sum())),
                normalized_max=float(np.max(np.abs(error))),
                signed_volume_integral=float(np.sum(vol*div)))
