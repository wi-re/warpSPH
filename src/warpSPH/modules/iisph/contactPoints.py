"""Closest surface contacts used by the IISPH wall pressure law."""


import torch


def computeIISPHContactPoints(state, provider):
    """Return closest contacts and a device-side coverage-failure flag."""
    if provider is None:
        return None, torch.zeros((), dtype=torch.bool, device=state.positions.device)
    distance, normal, hit = provider.signed_distance(state.positions)
    points = state.positions - distance[:, None] * normal.to(state.positions)
    # Keep uncovered queries finite until the batched diagnostic checkpoint.
    points = torch.where(hit[:, None], points, state.positions)
    return points, ((state.kinds == 0) & ~hit).any()
