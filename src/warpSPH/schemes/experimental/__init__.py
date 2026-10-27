"""Experimental step functions that are not registered fluid schemes (the rudimentary acoustic core).

One module per scheme, each a single per-step function wired by `schemes/builder.py`. Nothing is re-exported here: the
modules are imported by their full path (`warpSPH.schemes.experimental.<module>`) so that patching a name in the module patches the step.
"""
