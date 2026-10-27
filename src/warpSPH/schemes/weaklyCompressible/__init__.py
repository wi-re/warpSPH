"""Weakly compressible schemes: delta-SPH (and its delta+ / Sun 2017 variant through the scheme config).

One module per scheme, each a single per-step function wired by `schemes/builder.py`. Nothing is re-exported here: the
modules are imported by their full path (`warpSPH.schemes.weaklyCompressible.<module>`) so that patching a name in the module patches the step.
"""
