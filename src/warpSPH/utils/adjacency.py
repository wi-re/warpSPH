"""Explicit COO compatibility access; ordinary operators consume CSR directly."""
def cooRows(adjacency):
    """Use stored rows when present, otherwise request an uncached temporary."""
    rows = adjacency.paddedI
    return rows if rows is not None else adjacency.cooRows()
