"""Periodic 2D/3D compressible scale budgets on uniform observation grids.

Aluie (2013), https://arxiv.org/html/1012.5877, equations9 and13--19.
Positive Pi+Lambda denotes transfer out of resolved kinetic energy. Rates
are spatial means per volume; multiply by box volume for total powers.
Array axis j is coordinate j (ij ordering), unlike the older xy 2D adapter.
This is offline NumPy analysis, with no change to any evolution scheme.
"""
from itertools import product
import numpy as np


def _finite(value, name):
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError(f'{name} must be finite')
    return value


class PeriodicBudget:
    """Gaussian coarse-graining and spectral derivatives in a periodic box.

    ell is the Gaussian standard deviation in physical length units, not a
    sharp spectral cutoff. Products use grid collocation; no de-aliasing or
    closure is silently applied. Check grid sensitivity and identity residual.
    Real derivatives discard the even-grid Nyquist mode on their own axis.
    """
    def __init__(self, shape, length=1.):
        self.shape = tuple(shape)
        self.dim = len(self.shape)
        if self.dim not in (2, 3) or any(type(n) is not int or n < 8 for n in self.shape):
            raise ValueError('need a 2D/3D grid with integer extents >=8')
        self.length = np.broadcast_to(_finite(length, 'length'), (self.dim,)).copy()
        if np.any(self.length <= 0):
            raise ValueError('length must be positive')
        self.axes = tuple(range(self.dim))
        self.volume = float(np.prod(self.length))
        self.k2 = np.zeros(self.shape)
        self.k = []
        for axis, (n, length) in enumerate(zip(self.shape, self.length)):
            k = 2*np.pi*np.fft.fftfreq(n, d=length/n)
            shaped = [1]*self.dim; shaped[axis] = n
            self.k2 += k.reshape(shaped)**2
            if n % 2 == 0:
                k[n//2] = 0.
            self.k.append(k.reshape(shaped))

    def _grid(self, value, name, trailing=()):
        value = _finite(value, name)
        if value.shape != self.shape+trailing:
            raise ValueError(f'{name} must have shape {self.shape+trailing}')
        return value

    def filter(self, value, ell):
        value = _finite(value, 'field')
        if value.shape[:self.dim] != self.shape or value.ndim < self.dim:
            raise ValueError('field must start with the spatial grid shape')
        if not np.isfinite(ell) or ell < 0:
            raise ValueError('ell must be finite and nonnegative')
        if ell == 0:
            return value.copy()
        kernel = np.exp(-.5*ell*ell*self.k2).reshape(self.shape+(1,)*(value.ndim-self.dim))
        return np.fft.ifftn(np.fft.fftn(value, axes=self.axes)*kernel, axes=self.axes).real

    def derivative(self, value, axis):
        value = _finite(value, 'field')
        if axis not in self.axes or value.shape[:self.dim] != self.shape:
            raise ValueError('invalid derivative axis or field shape')
        wave = self.k[axis].reshape(self.k[axis].shape+(1,)*(value.ndim-self.dim))
        return np.fft.ifftn(1j*wave*np.fft.fftn(value, axes=self.axes), axes=self.axes).real

    def inviscidRHS(self, density, velocity, pressure, forceDensity=None, stress=None):
        """Euler collocation RHS, optionally including a supplied viscous stress.

        Used for algebra/aliasing checks; this is not the native solver RHS.
        Stress convention is momentum source +div(sigma), including any bulk
        term chosen by the caller. The diagnostic does not choose a viscosity.
        """
        rho = self._grid(density, 'density')
        v = self._grid(velocity, 'velocity', (self.dim,))
        p = self._grid(pressure, 'pressure')
        if np.any(rho <= 0):
            raise ValueError('density must be positive')
        force = np.zeros_like(v) if forceDensity is None else self._grid(forceDensity, 'force density', (self.dim,))
        sigma = None if stress is None else self._grid(stress, 'stress', (self.dim, self.dim))
        drho = np.zeros_like(rho)
        dmom = force.copy()
        for j in range(self.dim):
            drho -= self.derivative(rho*v[..., j], j)
            for i in range(self.dim):
                dmom[..., i] -= self.derivative(rho*v[..., i]*v[..., j], j)
                if sigma is not None:
                    dmom[..., i] += self.derivative(sigma[..., i, j], j)
        for i in range(self.dim):
            dmom[..., i] -= self.derivative(p, i)
        return drho, dmom

    def kineticRate(self, density, velocity, densityRate, momentumRate, ell):
        rho = self._grid(density, 'density')
        v = self._grid(velocity, 'velocity', (self.dim,))
        drho = self._grid(densityRate, 'density rate')
        dmom = self._grid(momentumRate, 'momentum rate', (self.dim,))
        rb = self.filter(rho, ell)
        if np.any(rb <= 0):
            raise ValueError('filtered density must be positive')
        favre = self.filter(rho[..., None]*v, ell)/rb[..., None]
        rate = np.sum(favre*self.filter(dmom, ell), axis=-1)-.5*np.sum(favre*favre, axis=-1)*self.filter(drho, ell)
        return float(rate.mean())

    def measure(self, density, velocity, pressure, *, ell, forceDensity=None,
                stress=None, observationMomentumFlux=None, returnFields=False):
        rho = self._grid(density, 'density')
        v = self._grid(velocity, 'velocity', (self.dim,))
        p = self._grid(pressure, 'pressure')
        if np.any(rho <= 0) or np.any(p <= 0):
            raise ValueError('density and pressure must be positive')
        force = np.zeros_like(v) if forceDensity is None else self._grid(forceDensity, 'force density', (self.dim,))
        sigma = None if stress is None else self._grid(stress, 'stress', (self.dim, self.dim))
        rb = self.filter(rho, ell)
        if np.any(rb <= 0):
            raise ValueError('filtered density must be positive; refine the observation grid/filter')
        mb = self.filter(rho[..., None]*v, ell)
        ub = self.filter(v, ell)
        favre = mb/rb[..., None]
        pb = self.filter(p, ell)
        pi = np.zeros_like(rho); lam = np.zeros_like(rho)
        dilation = np.zeros_like(rho); dissipation = np.zeros_like(rho)
        sb = None if sigma is None else self.filter(sigma, ell)
        for j in range(self.dim):
            lam += self.derivative(pb, j)*(favre[..., j]-ub[..., j])
            dilation += self.derivative(ub[..., j], j)
            for i in range(self.dim):
                gradient = self.derivative(favre[..., i], j)
                covariance = self.filter(rho*v[..., i]*v[..., j], ell)-mb[..., i]*favre[..., j]
                pi -= gradient*covariance
                if sb is not None:
                    dissipation += gradient*sb[..., i, j]
        fields = dict(kineticEnergyDensity=.5*rb*np.sum(favre*favre, axis=-1),
                      deformationTransfer=pi, baropycnalTransfer=lam,
                      pressureDilatation=pb*dilation,
                      forcingWork=np.sum(favre*self.filter(force, ell), axis=-1),
                      viscousWork=dissipation)
        fields['totalTransfer'] = pi+lam
        fields['predictedRate'] = -fields['totalTransfer']+fields['pressureDilatation']+fields['forcingWork']-dissipation
        if observationMomentumFlux is not None:
            # Particle observation flux C_ij=sum(m*v_i*v_j*W)/cellVolume.
            # The reconstructed momentum equation has -div(C-rho*v*v).
            # Its periodic kinetic work is +grad(v_favre):G(C-rho*v*v).
            flux = self._grid(observationMomentumFlux, 'observation momentum flux', (self.dim, self.dim))
            tau = flux-rho[..., None, None]*v[..., :, None]*v[..., None, :]
            filteredTau = self.filter(tau, ell)
            observationWork = np.zeros_like(rho)
            for i in range(self.dim):
                for j in range(self.dim):
                    observationWork += self.derivative(favre[..., i], j)*filteredTau[..., i, j]
            fields['observationStressWork'] = observationWork
            fields['predictedObservedRate'] = fields['predictedRate']+observationWork
        result = {key: float(value.mean()) for key, value in fields.items()}
        result.update(ell=float(ell), minFilteredDensity=float(rb.min()), viscousStressProvided=sigma is not None)
        if returnFields:
            result['fields'] = fields
        return result



class AcceptedSampleClock:
    """Fixed requested physical clock observing accepted states, never stepping.

    A late accepted state yields one real observation and counts missed targets;
    it never duplicates/interpolates states or changes the solver timestep.
    Repeated-time queries are idempotent. Output indices are separate from ticks.
    """
    def __init__(self, interval):
        import math
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError('sample interval must be finite and positive')
        self.interval = float(interval)
        self.nextTick = 0
        self.count = 0
        self.missedTargets = 0
        self.lastTime = -1.

    def due(self, time):
        import math
        time = float(time)
        if not math.isfinite(time) or time < 0 or time < self.lastTime:
            raise ValueError('accepted sample times must be finite, nonnegative and chronological')
        self.lastTime = time
        epsilon = min(1e-10, self.interval*1e-6)
        if time+epsilon < self.nextTick*self.interval:
            return False
        tick = math.floor((time+epsilon)/self.interval)
        self.missedTargets += max(0, tick-self.nextTick)
        self.nextTick = tick+1
        self.count += 1
        return True


def depositConservative(positions, mass, velocity, density, pressure, internalEnergy,
                        acceleration, *, gridSize, length=1., radius, heatRate=None):
    """Conservative compact-Wendland observation map, coordinate axes in ij order.

    One constant radius is used for all particles/frames. The per-particle
    discrete stencil is normalized, preserving extensive mass, momentum,
    internal energy, pressure-volume integral and forcing work. It is an
    observation filter, not a hydrodynamic kernel. Preserve its unresolved
    kinetic energy, second velocity moments and source-work defects rather than assigning them to heat.
    Supports 2D/3D cubic boxes; chunking bounds temporary candidate-pair storage.
    """
    x = _finite(positions, 'positions'); v = _finite(velocity, 'velocity')
    if x.ndim != 2 or x.shape[1] not in (2, 3) or not len(x) or v.shape != x.shape:
        raise ValueError('need N x dim positions/velocity with dim2 or3')
    count, dim = x.shape
    scalars = [_finite(a, name) for a, name in [(mass, 'mass'), (density, 'density'),
               (pressure, 'pressure'), (internalEnergy, 'internal energy')]]
    m, rho, p, u = scalars
    if any(a.shape != (count,) or np.any(a <= 0) for a in scalars):
        raise ValueError('mass, density, pressure and internal energy must be positive N vectors')
    a = _finite(acceleration, 'acceleration')
    heat = np.zeros(count) if heatRate is None else _finite(heatRate, 'heat rate')
    if a.shape != x.shape or heat.shape != m.shape:
        raise ValueError('acceleration/heat rate shapes do not match particles')
    if type(gridSize) is not int or gridSize < 8 or not np.isfinite(length) or length <= 0:
        raise ValueError('need integer gridSize >=8 and positive length')
    dx = length/gridSize
    if not np.isfinite(radius) or not 2*dx <= radius <= length/2:
        raise ValueError('constant observation radius must be between2 grid cells and half a box')
    extensive = np.column_stack((m, m[:, None]*v, m*p/rho, m*u,
                                m[:, None]*a, .5*m*np.sum(v*v, axis=1),
                                m*np.sum(v*a, axis=1), m*heat,
                                (m[:, None, None]*v[:, :, None]*v[:, None, :]).reshape(count, dim*dim)))
    channels = extensive.shape[1]
    totals = np.zeros((gridSize**dim, channels))
    cell = np.remainder(x, length)/dx-.5
    extent = int(np.ceil(radius/dx)); candidates = (2*extent+1)**dim
    if candidates > 1000000:
        raise ValueError('observation stencil too large; use a smaller radius or coarser grid')
    offsets = np.array(list(product(range(-extent, extent+1), repeat=dim)), dtype=np.int64)
    chunk = max(1, min(256, 262144//len(offsets)))
    strides = gridSize**np.arange(dim-1, -1, -1)
    for start in range(0, count, chunk):
        sl = slice(start, start+chunk)
        lower = np.floor(cell[sl]).astype(np.int64); fraction = cell[sl]-lower
        q = np.linalg.norm((offsets[None, :, :]-fraction[:, None, :])*dx, axis=-1)/radius
        weight = np.maximum(0., 1-q)**4*(1+4*q)
        normalizer = weight.sum(axis=1, keepdims=True)
        if np.any(normalizer <= 0):
            raise ValueError('empty particle observation stencil')
        weight /= normalizer
        cells = (lower[:, None, :]+offsets[None, :, :]) % gridSize
        index = np.sum(cells*strides, axis=-1).ravel()
        for component in range(channels):
            totals[:, component] += np.bincount(index, weights=(weight*extensive[sl, component, None]).ravel(), minlength=gridSize**dim)
    grid = totals.reshape((gridSize,)*dim+(channels,))/dx**dim
    occupied = grid[..., 0] > 0
    vel = np.full((gridSize,)*dim+(dim,), np.nan)
    np.divide(grid[..., 1:dim+1], grid[..., :1], out=vel, where=occupied[..., None])
    force = grid[..., dim+3:2*dim+3]
    cellVolume = dx**dim
    resolvedK = float((.5*grid[..., 0][occupied]*np.sum(vel[occupied]**2, axis=-1)).sum()*cellVolume)
    particleK = float(extensive[:, 2*dim+3].sum())
    sourcePower = float(extensive[:, 2*dim+4].sum())
    resolvedPower = float(np.sum(vel[occupied]*force[occupied])*cellVolume)
    conservation = totals.sum(axis=0)-extensive.sum(axis=0)
    return dict(density=grid[..., 0], velocity=vel, pressure=grid[..., dim+1],
        internalEnergyDensity=grid[..., dim+2], forceDensity=force,
        sourceWorkDensity=grid[..., 2*dim+4], heatDensity=grid[..., 2*dim+5],
        momentumFlux=grid[..., 2*dim+6:].reshape((gridSize,)*dim+(dim,dim)),
        metadata=dict(dim=dim, gridSize=gridSize, radius=float(radius), emptyCellFraction=float(1-occupied.mean()),
            relativeMassError=float(conservation[0]/m.sum()), momentumError=conservation[1:dim+1].tolist(),
            maxExtensiveConservationError=float(abs(conservation).max()),
            particleKineticEnergy=particleK, resolvedKineticEnergy=resolvedK, unresolvedKineticEnergy=particleK-resolvedK,
            internalEnergy=float(np.sum(m*u)), sourcePower=sourcePower, resolvedSourcePower=resolvedPower,
            unresolvedSourcePower=sourcePower-resolvedPower, heatPower=float(np.sum(m*heat))))
