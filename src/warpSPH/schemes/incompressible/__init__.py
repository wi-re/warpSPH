"""Projection scheme compositions; reusable equations live in modules.

The full-map DFSPH reference, IISPH, omni and Band compositions select the
package ProjectedStep driver. divergenceFree/VD+PS retains an ordinary RHS
and the selected integrator. Artificial compressibility remains a separate,
deferred dual-time method. Import scheme modules explicitly; the legacy
private names and experiment switches are compatibility interfaces.
"""
