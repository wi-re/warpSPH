"""Shared projection stages with force and boundary choices stated by callers."""

import torch


def carryProjectionFields(destination, source):
    """Copy the advanced map's fields without inventing warm-start semantics."""
    for name in (
        "positions",
        "velocities",
        "densities",
        "supports",
        "pressures",
        "soundspeeds",
        "surfaceIndicators",
        "surfaceNormals",
        "surfaceLambdas",
    ):
        value = getattr(source, name, None)
        if value is None:
            continue
        target = getattr(destination, name, None)
        if target is not None and torch.is_tensor(target) and target.shape == value.shape:
            target.copy_(value)
        else:
            setattr(destination, name, value.clone() if torch.is_tensor(value) else value)


def requireProjectedStep(system, config, schemeName):
    """Refuse a full-map callback under a multistage legacy ODE driver."""
    if getattr(system, "_projectedStepActive", False):
        return
    from warpSPHIntegrators.schemes import IntegrationSchemeType as I

    if getattr(config, "integrationScheme", None) in (
        I.forwardEuler,
        I.explicitEuler,
        I.semiImplicitEuler,
    ):
        return
    raise ValueError(
        f"{schemeName} advances a full projected state map; use the package ProjectedStep driver "
        "or a legacy single-evaluation Euler driver, rather than repeating it as an ODE RHS"
    )
