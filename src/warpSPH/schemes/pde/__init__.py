"""Schemes for general PDEs that are not fluid schemes (the wave equation).

One module per scheme, each a single per-step function wired by `schemes/builder.py`. Nothing is re-exported here: the
modules are imported by their full path (`warpSPH.schemes.pde.<module>`) so that patching a name in the module patches the step.
"""
