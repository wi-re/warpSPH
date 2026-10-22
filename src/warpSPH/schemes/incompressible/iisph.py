"""Single density-projection variant sharing the local DFSPH spatial stages.

This is the historical composed variant, not a literal implementation of
SPlisHSPlasH's separate IISPH coefficients. See the source inventory.
"""


def iisph_step(system, dt, config, schemeConfig, verbose=False):
    from .dfsphReference import dfsphReference_step

    return dfsphReference_step(
        system, dt, config, schemeConfig, verbose=verbose, skipDivergence=True
    )
