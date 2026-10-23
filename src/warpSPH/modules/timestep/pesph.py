"""PESPH CFL using the existing pair signal-speed kernel, including approach.

Apply physical limits to fixed requests as well as adaptive steps. Persist
the last accepted dt on the state so restart keeps the same growth cap.
"""

import torch
from warpSPHCore import SupportScheme, sphKernel_xi

from ..shockCapturing.CullenDehnen2010 import compute_vsig


def pesphTimestep(ctx, system):
    config, scheme = ctx.config, ctx.schemeConfig
    state = system.state
    signal = compute_vsig(state, config, scheme, SupportScheme.SuperSymmetric, system.adjacency)
    signal = torch.maximum(signal, state.soundspeeds)
    xi = float(sphKernel_xi(config.kernel.value, config.dim))
    limit = (config.cflFactor*state.supports/(xi*signal)).min()
    if config.maxDt is not None:
        limit = limit.clamp_max(config.maxDt)
    if config.adaptiveDt:
        if state.previousTimestep is not None:
            limit = torch.minimum(limit, torch.as_tensor(state.previousTimestep, device=limit.device,
                                                         dtype=limit.dtype)*config.dtGrowthFactor)
    else:
        if ctx.spec.dt is None:
            raise ValueError('PESPH fixed timesteps require an explicit dt')
        limit = limit.clamp_max(ctx.spec.dt)
    torch._assert_async(torch.isfinite(limit) & (limit > 0) & (limit >= config.minDt),
                        'PESPH physical timestep is non-finite or below minDt')
    return limit
