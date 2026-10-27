"""Compressible SPH cases (shock tubes, blasts, vortices, instabilities); `common` holds their shared settings."""


def __getattr__(name):
    """Back-compat: `warpSPH.cases.compressible` used to be the helper module (now `.common`)."""
    if name == 'common' or name.startswith('__'):
        raise AttributeError(name)
    import importlib
    try:
        return getattr(importlib.import_module('.common', __name__), name)
    except AttributeError:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}') from None
