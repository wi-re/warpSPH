"""Periodic Fourier acceleration driven by a reproducible OU process.

OU knots use a fixed physical clock and linear interpolation between knots.
Queries never consume random numbers as a function of RHS evaluation count:
missing knots are generated in chronological order, and previous ones retained.
The interpolation is a declared finite-clock approximation to continuous OU.
"""
from itertools import product
import math
import torch

__all__ = ['FourierOU']


class FourierOU:
    def __init__(self, *, dim=2, length=1., kMin=1, kMax=2, accelerationRMS=.1,
                 correlationTime=1., updateInterval=.125, solenoidalWeight=1., seed=1234):
        if dim not in (2, 3):
            raise ValueError('FourierOU dim must be 2 or 3')
        if type(kMin) is not int or type(kMax) is not int or not 1 <= kMin <= kMax <= 16:
            raise ValueError('FourierOU needs integer modes 1 <= kMin <= kMax <= 16')
        for name, value in [('length', length), ('correlationTime', correlationTime), ('updateInterval', updateInterval)]:
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'FourierOU {name} must be finite and positive')
        if not math.isfinite(accelerationRMS) or accelerationRMS < 0:
            raise ValueError('FourierOU accelerationRMS must be finite and nonnegative')
        if not math.isfinite(solenoidalWeight) or not 0 <= solenoidalWeight <= 1:
            raise ValueError('FourierOU solenoidalWeight must be in [0, 1]')
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError('FourierOU seed must be an integer in [0, 2**63)')
        self.parameters = dict(dim=dim, length=length, kMin=kMin, kMax=kMax,
            accelerationRMS=accelerationRMS, correlationTime=correlationTime,
            updateInterval=updateInterval, solenoidalWeight=solenoidalWeight, seed=seed)
        modes = [v for v in product(range(-kMax, kMax+1), repeat=dim)
                 if kMin*kMin <= sum(k*k for k in v) <= kMax*kMax
                 and next(k for k in v if k != 0) > 0]
        self.modes = torch.tensor(modes, dtype=torch.float64)
        normal = self.modes / self.modes.norm(dim=1, keepdim=True)
        longitudinal = normal[:, :, None] * normal[:, None, :]
        self.projection = solenoidalWeight*(torch.eye(dim)-longitudinal) + (1-solenoidalWeight)*longitudinal
        variance = solenoidalWeight**2*(dim-1) + (1-solenoidalWeight)**2
        self.scale = accelerationRMS / math.sqrt(len(modes)*variance)
        self.decay = math.exp(-updateInterval/correlationTime)
        self.innovation = math.sqrt(-math.expm1(-2*updateInterval/correlationTime))
        self.generator = torch.Generator(device='cpu').manual_seed(seed)
        self.knots = [self._draw()]
        self._deviceModes = {}
        self._boundCache = None

    def _draw(self):
        return torch.randn((2, len(self.modes), self.parameters['dim']),
                           generator=self.generator, dtype=torch.float64)

    def coefficients(self, time):
        time = float(time)
        if not math.isfinite(time) or time < 0:
            raise ValueError('FourierOU time must be finite and nonnegative')
        clock = time/self.parameters['updateInterval']
        index = math.floor(clock)
        while len(self.knots) <= index+1:
            self.knots.append(self.decay*self.knots[-1] + self.innovation*self._draw())
        raw = (1-(clock-index))*self.knots[index] + (clock-index)*self.knots[index+1]
        return self.scale*torch.einsum('mij,pmj->pmi', self.projection, raw)

    def accelerationBound(self, time):
        """Spatial norm bound over the next fixed update interval.

        Three enclosing knots bound both linear segments. Cache only this
        diagnostic bound; tensor identity/version checks retain correctness for
        prescribed-knot controls and checkpoint reconstruction.
        """
        time = float(time)
        if not math.isfinite(time) or time < 0:
            raise ValueError('FourierOU time must be finite and nonnegative')
        index = math.floor(time/self.parameters['updateInterval'])
        while len(self.knots) <= index+2:
            self.knots.append(self.decay*self.knots[-1] + self.innovation*self._draw())
        knots = self.knots[index:index+3]
        key = (index, self.scale, id(self.projection), self.projection._version,
               tuple((id(knot), knot._version) for knot in knots))
        if self._boundCache is None or self._boundCache[0] != key:
            coefficients = self.scale*torch.einsum('mij,kpmj->kpmi', self.projection, torch.stack(knots))
            bound = float(coefficients.norm(dim=-1).sum(dim=(1, 2)).max())
            self._boundCache = (key, bound)
        return self._boundCache[1]

    def __call__(self, positions, time):
        if positions.ndim != 2 or positions.shape[1] != self.parameters['dim']:
            raise ValueError('FourierOU positions must have shape (N, dim)')
        key = (positions.device, positions.dtype)
        if key not in self._deviceModes:
            self._deviceModes[key] = self.modes.to(device=positions.device, dtype=positions.dtype)*(2*math.pi/self.parameters['length'])
        wave = self._deviceModes[key]
        coeff = self.coefficients(time).to(device=positions.device, dtype=positions.dtype)
        acceleration = torch.zeros_like(positions)
        # Bound temporary phase storage to N x 8, independently of mode count.
        for start in range(0, len(wave), 8):
            phase = positions @ wave[start:start+8].T
            acceleration = acceleration + phase.cos() @ coeff[0, start:start+8] + phase.sin() @ coeff[1, start:start+8]
        return acceleration

    def state_dict(self):
        """Self-contained forcing checkpoint, including past-query reproducibility."""
        return dict(parameters=dict(self.parameters), knots=torch.stack(self.knots).clone(),
                    generator=self.generator.get_state().clone())

    @classmethod
    def from_state_dict(cls, state):
        field = cls(**state['parameters'])
        knots = torch.as_tensor(state['knots'], dtype=torch.float64, device='cpu')
        if knots.ndim != 4 or tuple(knots.shape[1:]) != tuple(field.knots[0].shape) or len(knots) < 1:
            raise ValueError('FourierOU checkpoint has incompatible knot shape')
        if not torch.isfinite(knots).all():
            raise ValueError('FourierOU checkpoint contains nonfinite knots')
        field.knots = list(knots.clone().unbind())
        field.generator.set_state(torch.as_tensor(state['generator'], dtype=torch.uint8, device='cpu'))
        return field

    def snapshot(self, time):
        """JSON-compatible applied field, for output interpretation rather than restart."""
        return dict(parameters=dict(self.parameters), time=float(time), modes=self.modes.tolist(),
                    coefficients=self.coefficients(time).tolist())
