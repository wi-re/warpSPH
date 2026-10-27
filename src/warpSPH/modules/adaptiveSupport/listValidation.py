"""Shared shape guard before speculative support solves on a Verlet list."""

def listFits(adjacency, particles) -> bool:
    """A list the solver may run on without validating it first: built for as many particles as there are now.
    (`buildVerletList` rebuilds on a changed particle count; a solve that skips the per-iteration check must not
    run a kernel on a list sized for other particles -- walls appended after the setup's list, for instance.)"""
    if adjacency is None or getattr(adjacency, 'queryPositions', None) is None or getattr(adjacency, 'referencePositions', None) is None:
        return False
    n = particles.positions.shape[0]
    return adjacency.queryPositions.shape[0] == n and adjacency.referencePositions.shape[0] == n and adjacency.numNeighbors.shape[0] == n
