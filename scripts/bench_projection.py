"""Sequential incompressible projection state, throughput and recorded-control probe.

Tiny A/B and throughput windows omit rendering; --video enables VisPy controls.
"""

import argparse
import ast
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time


def literal(value):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scheme",
        choices=("dfsph", "dfsphPredictor", "dfsphShift", "iisph"),
        required=True,
    )
    p.add_argument("--case", default="dambreak")
    p.add_argument("--nx", type=int, default=64)
    p.add_argument("--precision", choices=("float32", "float64"), default="float32")
    p.add_argument("--graph", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--profile", type=int, default=0)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--scheme-set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--end", type=float)
    p.add_argument("--video", action="store_true")
    p.add_argument("--daemon", action="store_true")
    p.add_argument("--plot-interval", type=int, default=50)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if min(a.steps, a.warmup) < 1 or a.profile < 0:
        p.error("steps/warmup must be positive and profile nonnegative")
    if a.end is not None and (a.end <= 0 or a.profile):
        p.error("--end must be positive and cannot profile")
    from warpSPHBootstrap import bootstrap

    bootstrap(precision=a.precision)
    import torch
    import warpSPH, warpSPHCore, warpSPHIntegrators
    from warpSPH.cases import importAll
    from warpSPH.runner import run
    from warpSPH.runner.case import getCase

    importAll()
    torch.manual_seed(1234)
    a.out.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__)]
    for package in (warpSPH, warpSPHCore, warpSPHIntegrators):
        sources += list(Path(package.__file__).parent.rglob("*.py"))
    (a.out / "sources.json").write_text(
        json.dumps({str(s): hashlib.sha256(s.read_bytes()).hexdigest() for s in sources}, indent=2)
        + "\n"
    )
    case = getCase(a.case)
    configure, post, initialize = case.configureScheme, case.postStep, case.initialConditions
    initial = [None]
    gaps, last, profiler = [], [None], [None]
    end = a.warmup + a.steps

    def configured(ctx):
        if configure:
            configure(ctx)
        for item in a.scheme_set:
            path, value = item.split("=", 1)
            keys = path.split(".")
            target = ctx.schemeConfig
            for key in keys[:-1]:
                target = getattr(target, key)
            setattr(target, keys[-1], literal(value))

    def initialized(ctx, system):
        if initialize:
            initialize(ctx, system)
        state = system.state
        initial[0] = {
            k: getattr(state, k).detach().clone()
            for k in ("positions", "velocities", "densities", "masses", "kinds")
        }

    def stepped(ctx, state, step):
        torch.cuda.synchronize()
        now = time.perf_counter()
        if last[0] is not None and a.warmup <= step < end:
            gaps.append(1000 * (now - last[0]))
        last[0] = now
        if post:
            post(ctx, state, step)
        if step % (100 if a.end else 20) == 0:
            print(f"PROBE step={step} t={float(state.t):.9g}", flush=True)
        if a.profile and step == end - 1:
            profiler[0] = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            )
            profiler[0].start()
        if a.profile and step == end + a.profile - 1:
            profiler[0].stop()
            profiler[0].export_chrome_trace(str(a.out / "trace.json"))

    case = replace(
        case, configureScheme=configured, initialConditions=initialized, postStep=stepped
    )
    values = {k: literal(v) for k, v in (item.split("=", 1) for item in a.set)}
    from dataclasses import fields
    from warpSPH.runner.caseSpec import CaseSpec

    specFields = {field.name for field in fields(CaseSpec)}
    overrides = {k: v for k, v in values.items() if k in specFields}
    caseParams = {k: v for k, v in values.items() if k not in specFields}
    if a.end:
        from warpSPH.modules.timestep import defaultTimestep

        timestep = case.timestep or defaultTimestep

        def clamped(ctx, state):
            return min(float(timestep(ctx, state)), max(0.0, a.end - float(state.t)))

        case = replace(case, timestep=clamped)
        overrides["tLimit"] = a.end
    publisher = None
    if a.daemon:
        sys.path.insert(0, "/home/lu26029/dev/warpSPHRemote/python")
        from warpsph_remote.publisher import SimulationPublisher

        publisher = SimulationPublisher(
            f"{a.scheme} {a.case} | {a.precision} | {a.out.name}",
            measurements=True,
            export_every_steps=100,
            autostart=False,
        )
        case = publisher.bind_case(case)
        (a.out / "daemon.json").write_text(json.dumps(dict(id=publisher.id), indent=2) + "\n")
    try:
        r = run(
            case,
            scheme=a.scheme,
            nx=a.nx,
            nSteps=None if a.end else end + a.profile,
            quiet=True,
            progress=True,
            cudaGraph=a.graph,
            plot=a.video,
            video=a.video,
            params=caseParams,
            plotBackend="vispy",
            show=False,
            exportRoot=str(a.out / "video"),
            plotInterval=a.plot_interval,
            store=a.daemon,
            storeMode="trajectory",
            velocityAlarmPlotInterval=1,
            stallProgress=1.0e-3,
            stallDtSteps=100,
            **overrides,
        )
        if publisher:
            publisher.recording(r.ctx)
            publisher.finish(r.ctx, r.state, r.nSteps)
            publisher.update(
                step=r.nSteps,
                time=float(r.state.t),
                dt=float(r.ctx.config.dt),
                particles=len(r.state.state.positions),
            )
            publisher.close("failed" if r.diverged else "completed", r.stopReason or "")
    except BaseException as error:
        if publisher:
            publisher.close("failed", str(error))
        raise
    state = r.state.state
    fields = {
        k: v.detach().cpu()
        for k, v in vars(state).items()
        if isinstance(v, torch.Tensor) and not k.startswith("_")
    }
    torch.save(fields, a.out / "state.pt")
    graph = getattr(r.ctx.schemeConfig, "_rhsGraph", None)
    stepGraph = r.ctx.scratch.get("stepGraph")
    rows = [row["stepTime_ms"] for row in r.trajectory if a.warmup < row["step"] <= end]
    fluid = state.kinds == 0
    data = dict(
        scheme=a.scheme,
        case=a.case,
        nx=a.nx,
        precision=a.precision,
        requested_graph=a.graph,
        particles=len(state.positions),
        steps=r.nSteps,
        time=float(r.state.t),
        diverged=bool(r.diverged),
        stop_reason=r.stopReason,
        wall_ms_mean=statistics.mean(gaps) if gaps else None,
        integrator_ms_mean=statistics.mean(rows) if rows else None,
        run_wall_seconds=r.wallTime,
        run_wall_ms_per_step=1000 * r.wallTime / r.nSteps,
        graph=(
            dict(disabled=graph.disabled, captures=graph.captures, replays=graph.replays)
            if graph
            else None
        ),
        step_graph=(
            dict(
                disabled=stepGraph.disabled, captures=stepGraph.captures, replays=stepGraph.replays
            )
            if stepGraph
            else None
        ),
        mass_relative_drift=float(
            (state.masses.double().sum() - initial[0]["masses"].double().sum())
            / initial[0]["masses"].double().sum()
        ),
        max_particle_mass_change=float((state.masses - initial[0]["masses"]).abs().max()),
        density_min=float(state.densities[fluid].min()),
        density_max=float(state.densities[fluid].max()),
        velocity_max=float(state.velocities[fluid].norm(dim=-1).max()),
        density_rms_relative=float(
            (
                (state.densities[fluid] / r.ctx.schemeConfig.fluid.restDensity - 1).square().mean()
            ).sqrt()
        ),
        kinetic_energy=float(
            (
                0.5
                * state.masses[fluid].double()
                * state.velocities[fluid].double().square().sum(-1)
            ).sum()
        ),
        initial_kinetic_energy=float(
            (
                0.5
                * initial[0]["masses"][fluid].double()
                * initial[0]["velocities"][fluid].double().square().sum(-1)
            ).sum()
        ),
        integrator=r.ctx.integrator.identifier.name,
        export_path=str(r.ctx.exportPath),
        daemon_run_id=publisher.id if publisher else None,
        gpu=torch.cuda.get_device_name(),
    )
    (a.out / "result.json").write_text(json.dumps(data, indent=2) + "\n")
    (a.out / "trajectory.json").write_text(json.dumps(r.trajectory) + "\n")
    print("RESULT " + json.dumps(data), flush=True)


if __name__ == "__main__":
    main()
