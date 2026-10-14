"""Compressible-gas schemes: Monaghan SPH, GSPH and Inutsuka GSPH, CompSPH, CRKSPH and the meshless MFM / MFV.

One module per scheme, each a single per-step function wired by `schemes/builder.py`. Nothing is re-exported here: the
modules are imported by their full path (`warpSPH.schemes.compressible.<module>`) so that patching a name in the module patches the step.
"""
