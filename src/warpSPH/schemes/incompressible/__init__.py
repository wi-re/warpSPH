"""Incompressible schemes: the projection / pressure-solve family (divergenceFree, the DFSPH and IISPH references, omniSPH, IISPH with pressure boundaries) and artificial-compressibility SPH (ACSPH).

One module per scheme, each a single per-step function wired by `schemes/builder.py`. Nothing is re-exported here: the
modules are imported by their full path (`warpSPH.schemes.incompressible.<module>`) so that patching a name in the module patches the step.
"""
