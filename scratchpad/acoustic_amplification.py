#!/usr/bin/env python
"""Per-step amplification of the WCSPH acoustic mode under each integrator.

The stiff oscillator in weakly-compressible SPH is the (rho, v) pair, not
(x, v):   drho/dt = -rho div(v),   dv/dt = -grad(p(rho))/rho.
Linearised for one Fourier mode, with w = c*k:

    drho/dt = -w*v ,   dv/dt = +w*rho

|A| > 1 means that mode grows every step. See DELTASPH_VALIDATION_PLAN.md 5.3.
"""
import math
import numpy as np

def amp(wdt):
    J = np.array([[0.0, -wdt], [wdt, 0.0]])          # the linearised pair
    out = {}

    # semiImplicitEuler (enum 21): x is staggered against v, but
    # applyQuantityUpdate() takes explicit_step(dt), so rho rides on the OLD
    # velocity -> the (rho, v) pair is plain explicit Euler.
    out['semiImplicitEuler (21)'] = np.eye(2) + J

    # What a genuinely staggered rho would give (rho evaluated at v^{n+1}).
    K = np.array([[1.0, 0.0], [wdt, 1.0]])           # v^{n+1} = v^n + wdt*rho^n
    D = np.array([[1.0, -wdt], [0.0, 1.0]])          # rho^{n+1} = rho^n - wdt*v^{n+1}
    out['  (staggered rho, for ref)'] = D @ K

    # symplecticEuler (enum 13): 2-stage kick-drift-kick (DualSPHysics). Both
    # v and rho advance by dt * k1, where k1 is evaluated on the half state
    # state^n + (dt/2)*k0 -- i.e. explicit midpoint on the pair.
    out['symplecticEuler (13)'] = np.eye(2) + J + J @ J / 2

    out['rungeKutta2 (1)'] = np.eye(2) + J + J @ J / 2
    out['rungeKutta4 (9)'] = sum(np.linalg.matrix_power(J, n) / math.factorial(n)
                                 for n in range(5))
    return out

for wdt in (0.05, 0.2, 0.5):
    print(f'w*dt = {wdt}')
    for k, M in amp(wdt).items():
        g = max(abs(np.linalg.eigvals(M)))
        print(f'   {k:28s} |A| = {g:.8f}   over 400 steps: {g**400:.4g}')
    # The Pade(1,1) map (2-eps)/(2+eps) is the Cayley transform: modulus
    # exactly 1 for imaginary eps, hence neutrally stable by construction.
    print(f'   {"Pade(1,1)/exp rho map":28s} |A| = 1.00000000   over 400 steps: 1')
    print()
