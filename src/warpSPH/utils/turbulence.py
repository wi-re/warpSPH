"""Offline diagnostics for periodic 2D particle flows (NumPy, FP64).

CIC is a stated measurement filter, not an extra hydrodynamic operation. Spectra
are of the deposited barycentric velocity, not a variable-density kinetic-energy
spectrum. No empty cell is filled or deconvolved. Particle PDFs bypass deposition.
"""
import numpy as np

__all__ = ['particleStatistics', 'weightedPDF', 'depositCIC2D', 'spectra2D',
           'structureFunctions2D', 'correlationTime', 'stationarityScreen']


def _positive(values, name):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError(f'{name} must be finite and positive')
    return values


def _moments(values, weights):
    mean = np.average(values, weights=weights)
    return float(mean), float(np.sqrt(np.average((values-mean)**2, weights=weights)))


def particleStatistics(mass, density, velocity, internalEnergy, *, gamma=5/3, length=1.):
    """Mass/volume quadratures; volume weights are m/rho, explicitly normalized.

    log density uses the physical box mean density sum(m)/L^2. RMS Mach uses
    local adiabatic sound speed, with the corresponding weighted bulk removed.
    """
    mass, density, u = (_positive(x, name) for x, name in
                        [(mass, 'mass'), (density, 'density'), (internalEnergy, 'internalEnergy')])
    velocity = np.asarray(velocity, dtype=np.float64)
    if mass.ndim != 1 or density.shape != mass.shape or u.shape != mass.shape or velocity.shape != (len(mass), 2):
        raise ValueError('expected N masses/densities/energies and N x 2 velocities')
    if not len(mass) or not np.isfinite(velocity).all() or not np.isfinite(gamma) or gamma <= 1 or not np.isfinite(length) or length <= 0:
        raise ValueError('need finite velocities, gamma > 1 and positive box length')
    volume = mass/density
    logrho = np.log(density/(mass.sum()/length**2))
    out = {'volumeQuadratureRelativeError': float(volume.sum()/length**2-1)}
    for name, weights in [('Mass', mass), ('Volume', volume)]:
        bulk = np.average(velocity, axis=0, weights=weights)
        speed2 = np.sum((velocity-bulk)**2, axis=1)
        out['rmsVelocity'+name] = float(np.sqrt(np.average(speed2, weights=weights)))
        out['rmsMach'+name] = float(np.sqrt(np.average(speed2/(gamma*(gamma-1)*u), weights=weights)))
        out['logDensityMean'+name], out['logDensityStd'+name] = _moments(logrho, weights)
        out['internalEnergyMean'+name], out['internalEnergyStd'+name] = _moments(u, weights)
        out['bulkVelocity'+name] = bulk.tolist()
    return out


def weightedPDF(values, weights, edges):
    """Probability *per bin*, with under/overflow kept in separate bins.

    Fixed common edges allow comparison between runs without silently
    renormalizing away tails. Boundary values follow numpy.histogram semantics.
    """
    values, weights, edges = map(lambda x: np.asarray(x, dtype=np.float64), (values, weights, edges))
    if values.ndim != 1 or weights.shape != values.shape or not len(values) or not np.isfinite(values).all():
        raise ValueError('values and weights must be nonempty finite vectors')
    if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError('PDF weights must be nonnegative with positive sum')
    if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
        raise ValueError('PDF edges must be finite and strictly increasing')
    bins = np.histogram(values, bins=edges, weights=weights)[0]
    return np.r_[weights[values < edges[0]].sum(), bins, weights[values > edges[-1]].sum()]/weights.sum()


def depositCIC2D(positions, mass, velocity, *, gridSize, length=1.):
    """Periodic cell-centred cloud-in-cell deposition of mass and momentum.

    Returns rho and barycentric velocity. Empty cells carry NaN velocity and
    invalidate spectra. The filter conserves integrated mass/momentum to FP64
    accumulation precision but generally removes unresolved kinetic energy.
    """
    positions, velocity = map(lambda x: np.asarray(x, dtype=np.float64), (positions, velocity))
    mass = _positive(mass, 'mass')
    if mass.ndim != 1 or not len(mass) or positions.shape != (len(mass), 2) or velocity.shape != positions.shape:
        raise ValueError('expected N x 2 positions/velocities and N masses')
    if not np.isfinite(positions).all() or not np.isfinite(velocity).all():
        raise ValueError('positions and velocities must be finite')
    if type(gridSize) is not int or gridSize < 4 or not np.isfinite(length) or length <= 0:
        raise ValueError('need integer gridSize >= 4 and positive length')
    cell = np.remainder(positions, length)*(gridSize/length)-.5
    lower = np.floor(cell).astype(np.int64)
    fraction = cell-lower
    extensive = np.column_stack((mass, mass[:, None]*velocity))
    sums = np.zeros((gridSize*gridSize, 3))
    for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        weight = (fraction[:, 0] if i else 1-fraction[:, 0])*(fraction[:, 1] if j else 1-fraction[:, 1])
        index = ((lower[:, 1]+j) % gridSize)*gridSize+(lower[:, 0]+i) % gridSize
        for component in range(3):
            sums[:, component] += np.bincount(index, weights=extensive[:, component]*weight, minlength=gridSize**2)
    sums = sums.reshape(gridSize, gridSize, 3)
    cellMass = sums[..., 0]
    occupied = cellMass > 0
    vel = np.full((gridSize, gridSize, 2), np.nan)
    np.divide(sums[..., 1:], cellMass[..., None], out=vel, where=occupied[..., None])
    bulk = np.average(velocity, weights=mass, axis=0)
    kinetic = .5*np.sum(mass*np.sum((velocity-bulk)**2, axis=1))
    resolvedKinetic = .5*np.sum(cellMass[occupied]*np.sum((vel[occupied]-bulk)**2, axis=1))
    return dict(density=cellMass/(length/gridSize)**2, velocity=vel,
                emptyCellFraction=float(1-occupied.mean()),
                relativeMassError=float(cellMass.sum()/mass.sum()-1),
                momentumError=(sums[..., 1:].sum(axis=(0, 1))-extensive[:, 1:].sum(axis=0)),
                unresolvedKineticFraction=float(1-resolvedKinetic/kinetic) if kinetic > 0 else 0.)


def spectra2D(velocity, density=None):
    """Full-FFT shell *sums*, E_v(k)=sum_shell |vhat|^2/2, vhat=fft(v)/N^2.

    k is the integer box wave number rounded to nearest radial shell. Remove
    volume-mean velocity. All shells through the diagonal Nyquist are retained
    for Parseval; use k <= N/4 for comparisons and still check filter sensitivity.
    Compressional/solenoidal projections apply to velocity. Density power is
    fft(rho/<rho>-1), without a half factor. No window or CIC deconvolution.
    """
    velocity = np.asarray(velocity, dtype=np.float64)
    if velocity.ndim != 3 or velocity.shape[0] != velocity.shape[1] or velocity.shape[-1] != 2 or not np.isfinite(velocity).all():
        raise ValueError('spectra require a fully covered finite N x N x 2 grid')
    n = velocity.shape[0]
    v = velocity-velocity.mean(axis=(0, 1))
    transform = np.fft.fftn(v, axes=(0, 1))/n**2
    k = np.fft.fftfreq(n)*n
    kx, ky = np.meshgrid(k, k, indexing='xy')
    k2 = kx*kx+ky*ky
    # Nyquist planes have ambiguous derivative sign for real discrete fields.
    # They are included in Parseval but must not be used for resolved slopes.
    dot = kx*transform[..., 0]+ky*transform[..., 1]
    longitudinal = np.divide(abs(dot)**2, k2, out=np.zeros_like(k2), where=k2 > 0)
    power = np.sum(abs(transform)**2, axis=-1)
    shells = np.floor(np.sqrt(k2)+.5).astype(int)
    def shell(values):
        return np.bincount(shells.ravel(), weights=values.ravel())
    out = dict(k=np.arange(shells.max()+1), velocityEnergy=.5*shell(power),
               compressionalEnergy=.5*shell(longitudinal),
               solenoidalEnergy=.5*shell(np.maximum(0., power-longitudinal)))
    out['parsevalError'] = float(out['velocityEnergy'].sum()-.5*np.mean(np.sum(v*v, axis=-1)))
    if density is not None:
        density = _positive(density, 'grid density')
        if density.shape != (n, n):
            raise ValueError('density must match velocity grid')
        rhohat = np.fft.fftn(density/density.mean()-1)/n**2
        out['densityPower'] = shell(abs(rhohat)**2)
    return out


def structureFunctions2D(velocity, *, length=1.):
    """Axis-averaged longitudinal/transverse second-order velocity increments.

    These are directional diagnostics, not assumed isotropic angular averages.
    """
    velocity = np.asarray(velocity, dtype=np.float64)
    if velocity.ndim != 3 or velocity.shape[0] != velocity.shape[1] or velocity.shape[-1] != 2 or not np.isfinite(velocity).all() or not np.isfinite(length) or length <= 0:
        raise ValueError('structure functions need a covered square vector grid and positive length')
    n = velocity.shape[0]
    longitudinal, transverse = [], []
    for offset in range(1, n//2+1):
        dx = np.roll(velocity, offset, axis=1)-velocity
        dy = np.roll(velocity, offset, axis=0)-velocity
        longitudinal.append(.5*(np.mean(dx[..., 0]**2)+np.mean(dy[..., 1]**2)))
        transverse.append(.5*(np.mean(dx[..., 1]**2)+np.mean(dy[..., 0]**2)))
    return dict(separation=length*np.arange(1, n//2+1)/n,
                longitudinal=np.array(longitudinal), transverse=np.array(transverse))


def correlationTime(times, values):
    """Scalar autocorrelation diagnostic after interpolation to a uniform clock.

    Biased FFT covariance, normalized at zero; integrate the initial positive
    lobe with the half-weight at lag zero. Inspect at most half the record.
    No crossing => right-censored estimate, not a certified decorrelation time.
    Strong drift can also make this diagnostic misleading; screen separately.
    """
    times, values = map(lambda x: np.asarray(x, dtype=np.float64), (times, values))
    if times.ndim != 1 or values.shape != times.shape or len(times) < 8 or not np.isfinite(times).all() or not np.isfinite(values).all() or np.any(np.diff(times) <= 0):
        raise ValueError('need >= 8 finite time samples with strictly increasing times')
    clock = np.linspace(times[0], times[-1], len(times))
    signal = np.interp(clock, times, values)
    signal -= signal.mean()
    variance = np.mean(signal**2)
    if variance <= np.finfo(float).eps**2*max(1., float(np.mean(values**2))):
        return dict(tau=None, effectiveSamples=None, censored=True, lag=np.array([0.]), acf=np.array([1.]))
    n = len(signal)
    power = abs(np.fft.rfft(signal, n=2*n))**2
    acf = np.fft.irfft(power, n=2*n)[:n//2+1]/(n*variance)
    crossings = np.flatnonzero(acf[1:] <= 0)
    stop = int(crossings[0]+1) if len(crossings) else len(acf)
    dt = clock[1]-clock[0]
    tau = float(dt*(.5+acf[1:stop].sum()))
    return dict(tau=tau, effectiveSamples=float((clock[-1]-clock[0])/(2*tau)),
                censored=not bool(len(crossings)), lag=np.arange(len(acf))*dt, acf=acf)


def stationarityScreen(times, values, *, blocks=(10., 20., 30., 40.), tolerance=.1):
    """Predeclared block-mean drift + correlation-length screen, not a proof.

    Time averages use piecewise-linear quadrature with interpolated endpoints,
    so adaptive accepted steps are not weighted as if uniformly spaced. Report
    passes only with <=tolerance spread, >=5 tau in the shortest block and an
    uncensored correlation estimate over the entire assessment window.
    """
    times, values, blocks = map(lambda x: np.asarray(x, dtype=np.float64), (times, values, blocks))
    if times.ndim != 1 or values.shape != times.shape or len(times) < 8 or not np.isfinite(times).all() or not np.isfinite(values).all() or np.any(np.diff(times) <= 0):
        raise ValueError('need finite increasing time series')
    if blocks.ndim != 1 or len(blocks) < 3 or not np.isfinite(blocks).all() or np.any(np.diff(blocks) <= 0) or blocks[0] < times[0] or blocks[-1] > times[-1] or not 0 < tolerance < 1:
        raise ValueError('need >=2 complete nonoverlapping blocks and tolerance in (0, 1)')
    means = []
    for low, high in zip(blocks[:-1], blocks[1:]):
        selected = (times > low) & (times < high)
        t = np.r_[low, times[selected], high]
        v = np.interp(t, times, values)
        means.append(float(np.trapezoid(v, t)/(high-low)))
    scale = max(abs(np.mean(means)), np.finfo(float).tiny)
    spread = float(np.ptp(means)/scale)
    selected = (times >= blocks[0]) & (times <= blocks[-1])
    corr = correlationTime(times[selected], values[selected])
    longEnough = corr['tau'] is not None and np.min(np.diff(blocks)) >= 5*corr['tau'] and not corr['censored']
    return dict(blockMeans=means, relativeSpread=spread, tau=corr['tau'], effectiveSamples=corr['effectiveSamples'],
                correlationCensored=corr['censored'], blockLongEnough=bool(longEnough),
                passed=bool(spread <= tolerance and longEnough))


def fieldCorrelationTime(times, fields):
    """Connected Eulerian field ACF, averaged over space/components.

    Subtract each component's temporal mean, interpolate actual sample times,
    and use the same initial-positive-lobe estimator as correlationTime. This
    tests spatial-pattern memory in addition to scalar global observables.
    """
    times, fields = np.asarray(times, dtype=np.float64), np.asarray(fields, dtype=np.float64)
    if times.ndim != 1 or fields.ndim < 2 or fields.shape[0] != len(times) or len(times) < 8 or not np.isfinite(times).all() or not np.isfinite(fields).all() or np.any(np.diff(times) <= 0):
        raise ValueError('need >=8 finite increasing times and fields with a matching leading axis')
    flat = fields.reshape(len(times), -1)
    if flat.shape[1] == 0:
        raise ValueError('field must have nonempty spatial axes')
    clock = np.linspace(times[0], times[-1], len(times))
    signal = np.stack([np.interp(clock, times, column) for column in flat.T], axis=1)
    signal -= signal.mean(axis=0)
    variance = np.mean(signal**2)
    if variance <= np.finfo(float).eps**2*max(1., float(np.mean(fields**2))):
        return dict(tau=None, effectiveSamples=None, censored=True, lag=np.array([0.]), acf=np.array([1.]))
    n = len(times)
    covariance = np.fft.irfft(abs(np.fft.rfft(signal, n=2*n, axis=0))**2, n=2*n, axis=0).mean(axis=1)
    acf = covariance[:n//2+1]/(n*variance)
    crossings = np.flatnonzero(acf[1:] <= 0)
    stop = int(crossings[0]+1) if len(crossings) else len(acf)
    dt = clock[1]-clock[0]
    tau = float(dt*(.5+acf[1:stop].sum()))
    return dict(tau=tau, effectiveSamples=float((clock[-1]-clock[0])/(2*tau)),
                censored=not bool(len(crossings)), lag=np.arange(len(acf))*dt, acf=acf)


__all__.append('fieldCorrelationTime')
