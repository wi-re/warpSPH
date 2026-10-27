"""Compressible SPH cases with solid walls; `common` holds the wall construction helpers."""


def __getattr__(name):
    """Back-compat: `warpSPH.cases.compressibleWalls` used to be the helper module (now `.common`)."""
    if name == 'common' or name.startswith('__'):
        raise AttributeError(name)
    import importlib
    try:
        return getattr(importlib.import_module('.common', __name__), name)
    except AttributeError:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}') from None
