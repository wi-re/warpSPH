"""Observational hooks for compare_lobovsky.py; no production solver edits."""
from pathlib import Path


def fluid_side_wall_diffusion(view, config, scheme, adjacency, renormalization, p):
    """Probe AC-2L wall flux using two copies of the fluid-side gradient."""
    from warpSPH.modules.density.gradRhoL import computeGradRhoL
    from warpSPH.modules.deltaSPH import computeScalarFieldDiffusion
    from warpSPH.enumTypes import DensityDiffusionScheme
    from warpSPHCore import (GradientScheme, OperationDirection, OperationProperties,
                             SupportScheme, WarpOperation, warpOperation)
    grad = computeGradRhoL(view, config, scheme, adjacency, renormalization, field=p)
    original = computeScalarFieldDiffusion(view, config, adjacency,
        DensityDiffusionScheme.deltaSPH, gradFieldL=grad, field=p)
    correction = warpOperation(view,
        OperationProperties(kernel=config.kernel, operation=WarpOperation.Divergence,
            supportMode=SupportScheme.SuperSymmetric,
            operationMode=OperationDirection.BoundaryToFluid,
            gradientMode=GradientScheme.Difference),
        queryValues=grad, domain=config.domain, adjacency=adjacency)
    return original + correction


def reference_wall_velocities(view, config, scheme, adjacency):
    """Static Lobovsky walls: fluid Shepard at the wall, particle-derived normal.

    Source noSlipHash3D + FREE_SLIP; intentionally scoped to this static fixture.
    It is a diagnostic alternative to mDBC's reflected-ghost sampling location.
    """
    import torch
    from warpSPHCore import (GradientScheme, OperationDirection, OperationProperties,
                             SupportScheme, WarpOperation, warpOperation)
    def op(operation, mode, values):
        return warpOperation(view,
            OperationProperties(kernel=config.kernel, operation=operation,
                supportMode=SupportScheme.SuperSymmetric, operationMode=mode,
                gradientMode=GradientScheme.Naive),
            queryValues=values, domain=config.domain, adjacency=adjacency)
    ones = torch.ones_like(view.pressures)
    den = op(WarpOperation.Interpolate, OperationDirection.FluidToBoundary, ones)
    average = op(WarpOperation.Interpolate, OperationDirection.FluidToBoundary, view.velocities)/den.clamp_min(1e-30)[:, None]
    normal = -op(WarpOperation.Gradient, OperationDirection.BoundaryToBoundary, ones)
    length = normal.norm(dim=-1)
    normal = torch.where((length > 1e-6)[:, None], normal/length.clamp_min(1e-30)[:, None], 0.)
    # Reflection is invariant under n -> -n, so the source's inward-normal
    # flip is unnecessary here. Its zero-normal wall value is exactly zero.
    velocity = average - 2*(average*normal).sum(-1)[:, None]*normal
    velocity = torch.where(((den > 0) & (length > 1e-6))[:, None], velocity, 0.)
    return torch.where((view.kinds == 1)[:, None], velocity, view.velocities)


def install_consistent_wall_velocity():
    """Static-fixture source closure at entry, every RK stage and shifting.

    The old source control patches only dualTime's imported alias. Preserve
    that historical control; this opt-in path also replaces the step-entry
    alias and refreshes wall velocities from the accepted fluid field before
    Michel shifting. Shared boundary implementations remain untouched.
    """
    import importlib
    from warpSPH.modules.artificialCompressible import dualTime
    from warpSPH.systems import artificialCompressible as system_module
    step_module = importlib.import_module('warpSPH.schemes.incompressible.artificialCompressible')
    step_module.computeBoundaryVelocities = reference_wall_velocities
    dualTime.computeBoundaryVelocities = reference_wall_velocities
    original_shift = system_module.solveShifting

    def shift(**kwargs):
        from warpSPHCore import buildVerletList, SupportScheme
        from warpSPH.configurations.compute import neighborSearchOptions
        state, config = kwargs['systemState'], kwargs['config']
        adjacency = buildVerletList(state, config.domain,
            verletScale=config.verletScale, supportMode=SupportScheme.SuperSymmetric,
            priorNeighborhood=kwargs['adjacency'], verbose=False, **neighborSearchOptions(config))
        state.velocities = reference_wall_velocities(state, config, kwargs['schemeConfig'], adjacency)
        return original_shift(**dict(kwargs, adjacency=adjacency))

    system_module.solveShifting = shift


def install(root, *, observe=True, wall_diffusion='native', acoustic_damper='off', reference_operators=None, rk_treatment='native', wall_velocity='native', shifting='warpsph-michel'):
    import numpy as np
    from warpSPH.modules.artificialCompressible import dualTime

    root = Path(root)
    if observe:
        root.mkdir()
    counter = {'iteration': -1, 'stage': 0, 'flags': None}
    original_iterate = dualTime.iterate
    original_surface = dualTime.surfaceAt
    original_residual = dualTime.spatialResidual

    def damper(view, config, adjacency, beta):
        from warpSPHCore import (GradientScheme, OperationDirection, OperationProperties,
                                 SupportScheme, WarpOperation, warpOperation)
        def op(operation, mode, values):
            return warpOperation(view,
                OperationProperties(kernel=config.kernel, operation=operation,
                    supportMode=SupportScheme.SuperSymmetric,
                    operationMode=OperationDirection.FluidToFluid, gradientMode=mode),
                queryValues=values, domain=config.domain, adjacency=adjacency)
        div = op(WarpOperation.Divergence, GradientScheme.Difference, view.velocities)
        grad = op(WarpOperation.Gradient, GradientScheme.Summation, div)
        return (beta * view.supports / 2).unsqueeze(-1) * grad

    def iterate(*args, **kwargs):
        counter['iteration'] += 1
        counter['stage'] = 0
        counter['bdf'] = args[0].bdf
        if rk_treatment == 'source':
            from dataclasses import replace
            from types import SimpleNamespace
            problem, *state = args
            tab = problem.tableau
            # Exact reference-code diagnostic, including integer (3/2)=1.
            # Its stage input factor belongs to the DESTINATION row; final
            # weights each have their own factor. This is not the paper's
            # BDF coefficient and must never become a silent default.
            ratio = 1. / problem.schemeConfig.acParams.dtOverDtau
            factor = 1. / (1. + tab.b * ratio)
            tableau = SimpleNamespace(a=tab.a*factor[:, None], b=tab.b*factor, c=tab.c)
            args = (replace(problem, tableau=tableau), *state)
        return original_iterate(*args, **kwargs)

    def surface(*args, **kwargs):
        flags = original_surface(*args, **kwargs)
        counter['flags'] = flags
        if shifting == 'dyngpusph-reference':
            import torch
            counter['stage0'] = args[1]
            if args[1].positions.is_cuda and torch.cuda.is_current_stream_capturing():
                counter['captured_stage0'] = args[1]
        return flags

    def residual(view, config, scheme, adjacency, k1, k2, diffusion, bodyForce, nu, **kwargs):
        if acoustic_damper == 'source' and scheme.acParams.fusedResidual:
            from warpSPHCore import (GradientScheme, OperationDirection, OperationProperties,
                                     SupportScheme, WarpOperation, warpOperation)
            from warpSPH.modules.artificialCompressible.residual import fusedResidualTerms
            divF=warpOperation(view,OperationProperties(kernel=config.kernel,
                operation=WarpOperation.Divergence,supportMode=SupportScheme.SuperSymmetric,
                operationMode=OperationDirection.FluidToFluid,gradientMode=GradientScheme.Difference),
                queryValues=view.velocities,domain=config.domain,adjacency=adjacency)
            div,pressure,visc,gradDiv=fusedResidualTerms(view,config,scheme,adjacency,nu,fluidDiv=divF,
                referenceVelocities=view.velocities if kwargs.get('preparedViscosity',False) else None)
            smoothing=diffusion if kwargs.get('taper') is None else diffusion*kwargs['taper']
            rP=-k1*view.densities*div+k2*smoothing
            rV=pressure+visc+bodyForce+(k1.sqrt()*view.supports/2)[:,None]*gradDiv
        else:
            rP, rV = original_residual(view, config, scheme, adjacency, k1, k2, diffusion, bodyForce, nu, **kwargs)
        if acoustic_damper == 'source' and not scheme.acParams.fusedResidual:
            rV = rV + damper(view, config, adjacency, k1.sqrt())
        m, s = counter['iteration'], counter['stage']
        counter['stage'] += 1
        if observe and m in (0, 1, 199):
            keep = view.kinds != 2
            arrays = dict(x=view.positions, v=view.velocities, p=view.pressures,
                          rho=view.densities, m=view.masses, bc=view.kinds,
                          k1=k1, k2=k2, nu=nu, diffusion=diffusion,
                          taper=kwargs['taper'], rP=rP, rV=rV,
                          lambdas=counter['flags'].lambdas)
            arrays = {key: value[keep].detach().cpu().numpy().copy() for key, value in arrays.items()}
            if m == 0 and s == 0:
                from warpSPH.modules.density.gradRhoL import computeGradRhoL
                grad = computeGradRhoL(view, config, scheme, adjacency,
                                      counter['flags'].renormalization, field=view.pressures)
                arrays['gradP'] = grad[keep].detach().cpu().numpy().copy()
                if reference_operators is not None:
                    import json
                    import torch
                    from analyze_operators import reference, stats
                    from warpSPH.modules.artificialCompressible.workingState import workingState
                    reports = {}; closures = {}
                    for rm, rs in ((0, 1), (0, 2), (199, 2)):
                        a = reference(Path(reference_operators), rm, rs)
                        x, v = view.positions.clone(), view.velocities.clone()
                        x[keep] = torch.as_tensor(a['x'], device=x.device)
                        v[keep] = torch.as_tensor(a['v'], device=v.device)
                        replay = workingState(view, x, v, view.pressures)
                        beta = torch.full_like(k1, 375. if rm < 2 else 187.5)
                        got = damper(replay, config, adjacency, beta)[keep].detach().cpu().numpy()
                        fluid = arrays['bc'] == 0
                        reports[f'i{rm:03d}-s{rs}'] = stats((got-a['terms'][:, 6:8])[fluid])
                        from warpSPHCore import (GradientScheme, OperationDirection, OperationProperties,
                                                 SupportScheme, WarpOperation, warpOperation)
                        from warpSPH.modules.mdbc import computeBoundaryVelocities
                        from warpSPH.modules.pressure import computePressureForceSurfaceAware
                        def divergence(velocities):
                            return warpOperation(replay,
                                OperationProperties(kernel=config.kernel, operation=WarpOperation.Divergence,
                                    supportMode=SupportScheme.SuperSymmetric,
                                    operationMode=OperationDirection.AllToAll,
                                    gradientMode=GradientScheme.Difference),
                                queryValues=velocities, domain=config.domain, adjacency=adjacency)[keep].detach().cpu().numpy()
                        local_wall = computeBoundaryVelocities(replay, config, scheme, adjacency)
                        local_div = divergence(local_wall)
                        source_wall = v.clone()
                        nv = torch.as_tensor(a['norm'], device=v.device)
                        tangent = torch.stack((nv[:, 1], -nv[:, 0]), -1)
                        average = torch.as_tensor(a['wallV'], device=v.device)
                        source_v = nv*(-(average*nv).sum(-1))[:, None] + tangent*((average*tangent).sum(-1))[:, None]
                        physical_v = source_wall[keep]
                        wall_rows = view.kinds[keep] == 1
                        physical_v[wall_rows] = source_v[wall_rows]
                        source_wall[keep] = physical_v
                        source_div = divergence(source_wall)
                        source_impl_div = divergence(reference_wall_velocities(replay, config, scheme, adjacency))
                        pressure = replay.pressures.clone()
                        pressure[keep] = torch.as_tensor(a['p'], device=v.device)
                        replay.pressures = pressure
                        pressure_force = computePressureForceSurfaceAware(replay, config, scheme, adjacency)[keep].detach().cpu().numpy()
                        closures[f'i{rm:03d}-s{rs}'] = {
                            'local_wall_divergence_error': stats((local_div-a['terms'][:, 8])[fluid]),
                            'source_wall_divergence_error': stats((source_div-a['terms'][:, 8])[fluid]),
                            'source_closure_implementation_error': stats((source_impl_div-a['terms'][:, 8])[fluid]),
                            'same_pressure_force_error': stats((pressure_force-a['terms'][:, :2])[fluid]),
                            'reference_divergence': stats(a['terms'][fluid, 8])}
                    (root/'damper-replay.json').write_text(json.dumps(reports, indent=2)+'\n')
                    (root/'closure-replay.json').write_text(json.dumps(closures, indent=2)+'\n')
            np.savez(root/f'i{m:03d}-s{s}.npz', **arrays)
            print(f'LOCAL AUDIT iteration={m} stage={s}', flush=True)
        return rP, rV

    dualTime.iterate = iterate
    dualTime.surfaceAt = surface
    dualTime.spatialResidual = residual
    if wall_velocity == 'source':
        dualTime.computeBoundaryVelocities = reference_wall_velocities
    elif wall_velocity == 'source-consistent':
        install_consistent_wall_velocity()
    if shifting == 'dyngpusph-reference':
        from warpSPH.systems import artificialCompressible as system_module
        from warpSPH.modules.artificialCompressible.dyngpusphShifting import shift_and_remap, reference_surface
        from warpSPHCore import buildVerletList, SupportScheme
        from warpSPH.configurations.compute import neighborSearchOptions
        import torch
        original_bdf_source = dualTime.bdfSource
        shift_count = [0]
        shift_log = root.parent/'reference-shift.jsonl'

        def source_bdf(problem, x, v):
            position, momentum = original_bdf_source(problem, x, v)
            prior = getattr(problem.schemeConfig, '_dyngpusphShiftBDF', None)
            if prior is not None:
                position = position-prior
            return position, momentum

        def source_shift(**kwargs):
            state, config, scheme = kwargs['systemState'], kwargs['config'], kwargs['schemeConfig']
            if scheme.acParams.bdfShiftCorrection:
                raise ValueError('Reference shift history requires bdfShiftCorrection=False')
            adjacency = buildVerletList(state,config.domain,verletScale=config.verletScale,
                supportMode=SupportScheme.SuperSymmetric,priorNeighborhood=kwargs['adjacency'],
                verbose=False,**neighborSearchOptions(config))
            state.velocities = reference_wall_velocities(state,config,scheme,adjacency)
            flags = counter['flags']
            if flags is None:
                raise RuntimeError('Reference shifting requires the last dual-iteration surface fields')
            before_v, before_p = state.velocities, state.pressures
            # Graph validation performs an eager call after capture, which
            # overwrites ordinary Python hooks. The captured geometry tensors
            # keep updating during replay; use those when the graph survives.
            captured_stage0 = counter.pop('captured_stage0',None)
            shift_stage0 = (captured_stage0 if captured_stage0 is not None and
                getattr(scheme,'_acsphGraphDisabled',None) is None else counter['stage0'])
            source_lambda,source_normals,source_edges = reference_surface(
                shift_stage0,state,config,accepted_adjacency=adjacency)
            shift, recorded, pressure, velocity, active, _, _, _ = shift_and_remap(
                state,config,dt=kwargs['dt'],lambdas=source_lambda,
                edges=source_edges,normals=source_normals,
                pair_multiplicity=scheme._dyngpusphShiftPairMultiplicity)
            previous = getattr(scheme,'_dyngpusphShift',torch.zeros_like(shift))
            old_bdf = getattr(scheme,'_dyngpusphShiftBDF',torch.zeros_like(shift))
            # Compiled source deliberately passes the previous shift twice to
            # variable-step BDF2; alpha+beta+gamma=0 gives this difference.
            new_bdf = counter['bdf'][0]*(recorded-previous)
            scheme._dyngpusphShiftBDF = torch.where((active>0)[:,None],new_bdf,old_bdf)
            scheme._dyngpusphShift = recorded
            state.pressures, state.velocities = pressure, velocity
            shift_count[0] += 1
            fluid = state.kinds==0
            row = dict(step=shift_count[0],maxShiftDx=float(shift[fluid].norm(dim=-1).max()/config.dx),
                maxPressureRemapPa=float((pressure-before_p)[fluid].abs().max()),
                maxVelocityRemap=float((velocity-before_v)[fluid].norm(dim=-1).max()),
                remapKineticEnergyChange=float((.5*state.masses[fluid]*(velocity[fluid].square().sum(-1)-before_v[fluid].square().sum(-1))).sum()),
                remapMomentumChange=(state.masses[fluid,None]*(velocity-before_v)[fluid]).sum(0).cpu().tolist())
            with shift_log.open('a') as log:
                log.write(__import__('json').dumps(row)+'\n')
            if shift_count[0] <= 3 or shift_count[0]%100==0:
                print('DYNGPUSPH SHIFT',row,flush=True)
            return shift

        dualTime.bdfSource = source_bdf
        system_module.solveShifting = source_shift
    if observe:
        from warpSPH.systems import artificialCompressible as system_module
        original_shift = system_module.solveShifting
        shift_count = [0]

        def shift(**kwargs):
            state = kwargs['systemState']
            keep = state.kinds != 2
            arrays = {k: getattr(state, k)[keep].detach().cpu().numpy().copy()
                      for k in ('positions', 'velocities', 'pressures')}
            dx = original_shift(**kwargs)
            arrays['shift'] = dx[keep].detach().cpu().numpy().copy()
            np.savez(root/f'shift-{shift_count[0]:03d}.npz', **arrays)
            shift_count[0] += 1
            return dx

        system_module.solveShifting = shift
    if rk_treatment == 'source':
        dualTime.pointImplicit = lambda problem, stage: 1.

    if wall_diffusion == 'source':
        # Diagnostic only: dynGPUSPH sph.cuh getContrib2 uses 2 gradP_i
        # rather than gradP_i+gradP_wall for wall neighbours in AC-2L.
        from warpSPH.modules.artificialCompressible import pressureSmoothing
        pressureSmoothing._biLaplacian = fluid_side_wall_diffusion
