"""Audit local reference pressure/factor scaling without advancing a simulation.

Mass and density scale together, preserving positions and apparent volumes.
The same physical-pressure unit probe should expose the true 1/rho scaling.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from warpSPHBootstrap import bootstrap

    bootstrap(precision=args.precision)
    import torch
    from warpSPH.cases import importAll
    from warpSPH.runner import getCase, prepareContext
    from warpSPH.runner.caseSpec import CaseSpec
    from warpSPH.modules.incompressible.operators import rebuildAdjacency, pressureAcceleration
    from warpSPH.modules.incompressible.reference import (
        computeReferenceFactor,
        referenceDensityRate,
    )
    from warpSPH.modules.density import computeDensities

    importAll()
    torch.manual_seed(1234)
    case = getCase("tgv")
    spec = (
        CaseSpec(caseName="tgv", scheme="dfsphReference", params=dict(case.params))
        .merged(**case.defaults)
        .merged(nx=24, scheme="dfsphReference")
    )
    ctx = prepareContext(case, spec)
    system = case.buildSystem(ctx)
    state = system.state
    adjacency = rebuildAdjacency(state, system, ctx.config)
    state.densities = computeDensities(state, ctx.config, ctx.schemeConfig, adjacency)
    mass, density = state.masses, state.densities
    rho0 = ctx.schemeConfig.fluid.restDensity
    fluid = state.kinds == 0
    p = torch.zeros_like(density)
    p[len(p) // 2] = 1.0
    rows = []
    for scale in (1.0, 1000.0):
        state.masses, state.densities = mass * scale, density * scale
        ctx.schemeConfig.fluid.restDensity = rho0 * scale
        factor = computeReferenceFactor(state, ctx.config, ctx.schemeConfig, adjacency)
        acceleration = pressureAcceleration(state, ctx.config, adjacency, p, fluid)
        operator = referenceDensityRate(
            state, ctx.config, ctx.schemeConfig, adjacency, acceleration
        ) / (rho0 * scale)
        i = len(p) // 2
        rows.append(
            dict(
                mass_density_scale=scale,
                factor=float(factor[i]),
                true_operator_diagonal=float(operator[i]),
                factor_over_diagonal=float(factor[i] / operator[i]),
            )
        )
    data = dict(
        precision=args.precision,
        particles=len(p),
        rows=rows,
        factor_ratio=rows[1]["factor"] / rows[0]["factor"],
        operator_diagonal_ratio=rows[1]["true_operator_diagonal"]
        / rows[0]["true_operator_diagonal"],
        verdict="Volume-only factor and physical-pressure diagonal have different density normalization; preserved local variant, no numerical change adopted.",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data), flush=True)


if __name__ == "__main__":
    main()
