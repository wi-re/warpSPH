"""Analytic incident shock and independent two-gamma ideal-gas star state.

The scalar reference solves the pressure matching equation with bracketed
root finding. It is for diagnostics/tests, never the production pair force.
"""
import math
from scipy.optimize import brentq

__all__ = ['incidentShock', 'interfaceStarState', 'weakShockReference']


def incidentShock(rho, pressure, gamma, mach):
    if rho <= 0 or pressure <= 0 or gamma <= 1 or mach < 1:
        raise ValueError('shock: require rho,p > 0, gamma > 1 and Mach >= 1')
    ratio = (gamma + 1) * mach**2 / (2 + (gamma - 1) * mach**2)
    speed = mach * math.sqrt(gamma * pressure / rho)
    return dict(density=rho * ratio, pressure=pressure * (1 + 2 * gamma / (gamma + 1) * (mach**2 - 1)),
                velocity=speed * (1 - 1 / ratio), speed=speed)


def interfaceStarState(left, right):
    """States are (density, velocity, pressure, gamma); return the contact state.

    Handles shocks/rarefactions on either side, rejecting vacuum solutions.
    """
    for r, v, p, g in (left, right):
        if not all(math.isfinite(x) for x in (r, v, p, g)) or r <= 0 or p <= 0 or g <= 1:
            raise ValueError('Riemann state: require finite rho,p > 0 and gamma > 1')

    def wave(p, state):
        r, v, pk, g = state
        a = math.sqrt(g * pk / r)
        if p > pk:
            return (p - pk) * math.sqrt(2 / ((g + 1) * r) / (p + (g - 1) / (g + 1) * pk))
        return 2 * a / (g - 1) * ((p / pk)**((g - 1) / (2 * g)) - 1)

    def residual(p):
        return wave(p, left) + wave(p, right) + right[1] - left[1]

    if residual(0) >= 0:
        raise ValueError('Riemann reference: vacuum solution is not supported')
    hi = max(left[2], right[2])
    for _ in range(100):
        if residual(hi) >= 0:
            break
        hi *= 2
    else:
        raise ValueError('Riemann reference: unable to bracket pressure')
    p = brentq(residual, 0, hi, xtol=1e-14, rtol=1e-14)
    velocity = .5 * (left[1] + right[1] + wave(p, right) - wave(p, left))

    def starDensity(state):
        r, _, pk, g = state
        ratio, beta = p / pk, (g - 1) / (g + 1)
        return r * (ratio + beta) / (beta * ratio + 1) if p > pk else r * ratio**(1/g)

    def waveSpeed(state, direction):
        r, v, pk, g = state
        c = math.sqrt(g * pk / r)
        factor = math.sqrt((g + 1) / (2 * g) * p / pk + (g - 1) / (2 * g)) if p > pk else 1.
        return v + direction * c * factor

    return dict(pressure=p, velocity=velocity, densityLeft=starDensity(left), densityRight=starDensity(right),
                waveLeft=waveSpeed(left, -1), waveRight=waveSpeed(right, 1))


def weakShockReference(rho0, rho1, pressure, gamma0, gamma1, mach, amplitude, wavelength=1.):
    """Irrotational, weak-shock linear estimate from post-shock tangential shear.

    Wouchuk–Nishihara's interface contribution, as written in Sano et al.
    (2012), ApJ 758, 126, Eqs. 6–9, DOI 10.1088/0004-637X/758/2/126.
    Coordinates are reflected relative to that paper: the incident shock here
    travels in +x, and the heavy/light interface is x=a*cos(k*y).

    This is NOT the full compressible Wouchuk series: bulk-vorticity/acoustic
    history corrections are omitted. Use for weak shocks and small amplitude,
    with a reflected shock; do not treat it as an exact nonlinear reference.
    Different ideal-gas gammas enter through their planar Riemann states.
    """
    if amplitude < 0 or wavelength <= 0:
        raise ValueError('reference: require amplitude >= 0 and wavelength > 0')
    incident = incidentShock(rho0, pressure, gamma0, mach)
    star = interfaceStarState((incident['density'], incident['velocity'], incident['pressure'], gamma0),
                              (rho1, 0., pressure, gamma1))
    if star['pressure'] < incident['pressure'] * (1-1e-12):
        raise ValueError('weak-shock reference currently requires a reflected shock')
    k = 2*math.pi/wavelength
    a = amplitude*wavelength
    reflectedRipple = a*(1-star['waveLeft']/incident['speed'])
    transmittedRipple = a*(1-star['waveRight']/incident['speed'])
    leftTangential = k*reflectedRipple*(star['velocity']-incident['velocity'])
    rightTangential = k*transmittedRipple*star['velocity']
    totalDensity = star['densityLeft']+star['densityRight']
    growth = (star['densityRight']*rightTangential-star['densityLeft']*leftTangential)/totalDensity
    # Integrate a sinusoidal vortex sheet over one half wavelength.
    circulation = 2*(rightTangential-leftTangential)/k
    atwood = (star['densityRight']-star['densityLeft'])/totalDensity
    compressedAmplitude = a*(1-star['velocity']/incident['speed'])
    return dict(model='weak-shock irrotational interface contribution; not full compressible linear theory',
        growthRate=growth, circulationUpper=circulation, circulationLower=-circulation,
        postShockAmplitude=compressedAmplitude, postShockAtwood=atwood,
        impulseGrowthRate=k*compressedAmplitude*atwood*star['velocity'],
        leftTangentialAmplitude=leftTangential, rightTangentialAmplitude=rightTangential,
        inverseTimeScale=k*abs(growth))
