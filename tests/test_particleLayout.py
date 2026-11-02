"""Independent Morton/CSR oracles and simulator-boundary layout transactions."""
from copy import copy
from dataclasses import replace

import h5py
import pytest
import torch

from warpSPH.evaluation import MortonStepper, adopt_spatial_layout, canonical_view
from warpSPH.evaluation.layout import (morton_order,
                                       permute_state, remap_adjacency)
from warpSPHIntegrators import get_scheme
from warpSPHIntegrators.solvers import JFNKSolver
from warpSPHCore import AdjacencyList, get_torch_precision, verletUsage
from warpSPH.utils.domain import buildDomainDescription
from test_spatialEvaluationResources import setup


def shuffled(method='CompSPH', *, compatible=True, adaptive=True, device='cpu'):
    ctx, system = setup(method, compatible=compatible, adaptive=adaptive,device=device)
    order = torch.randperm(system.state.UIDs.numel(), generator=torch.Generator().manual_seed(17))
    system.state = permute_state(system.state, order)
    system.adjacency = None
    return ctx, system


def particle_pairs(adjacency, ids):
    rows, cols = adjacency.livePairs()
    return sorted(zip(ids[rows.long()].tolist(), ids[cols.long()].tolist()))


@pytest.mark.parametrize('dim', [1, 2, 3])
@pytest.mark.parametrize('bits', [3,21])
def test_morton_keys_match_independent_interleaving_and_stable_periodic_ties(dim,bits):
    domain = buildDomainDescription(l=2, dim=dim, periodic=True, dtype=get_torch_precision())
    # Exact binary locations include periodic copies and repeated Morton cells.
    positions = torch.tensor([[-.75, .25, -.25], [.5, -.5, .5], [-.75, .25, -.25],
                              [1.25, -1.75, 1.75], [-1., 1., -1.], [.75, .75, .75]],
                             dtype=domain.min.dtype)[:, :dim].contiguous()
    def key(row):
        coords = [int((((float(x)+1)/2) % 1)*2**bits) for x in row]
        return sum(((c >> bit) & 1) << (3*bit+d) for d,c in enumerate(coords) for bit in range(bits))
    expected = sorted(range(len(positions)), key=lambda i:key(positions[i]))
    assert morton_order(positions, domain, bits=bits).tolist() == expected


@pytest.mark.parametrize('stored_i', [False, True])
@pytest.mark.parametrize('offset_dtype', [torch.int32, torch.int64])
def test_csr_remap_preserves_independent_query_reference_owners_anchors_and_padding(stored_i, offset_dtype):
    # N=3, E=3 deliberately: edge arrays must never be mistaken for particle data.
    counts = torch.tensor([2, 0, 1], dtype=torch.int32)
    offsets = torch.tensor([0, 2, 2], dtype=offset_dtype)
    rows = torch.tensor([0,0,2,-1,-1], dtype=torch.int32) if stored_i else None
    cols = torch.tensor([1,3,0,-1,-1], dtype=torch.int32)
    qp = torch.arange(3.).reshape(-1,1); rp = torch.arange(4.).reshape(-1,1)+10
    qh = torch.tensor([1.,2.,3.]); rh = torch.tensor([4.,5.,6.,7.])
    adj = AdjacencyList(rows, cols, counts, offsets, 3, 4, qp, rp, qh, rh, numEdges=3)
    p, r = torch.tensor([2,0,1]), torch.tensor([2,0,3,1])
    out, edge_order = remap_adjacency(adj, p, r, return_edge_order=True)
    assert out.numNeighbors.tolist() == [1,2,0] and out.edgeOffsets.tolist() == [0,1,3]
    assert out.paddedJ.tolist() == [1,3,2,-1,-1] and edge_order.tolist() == [2,0,1]
    assert out.paddedI is None if not stored_i else out.paddedI.tolist() == [0,1,1,-1,-1]
    assert out.edgeOffsets.dtype == offset_dtype and out.paddedJ.dtype == cols.dtype
    assert out.numEdges == 3 and out.hashMap is None
    torch.testing.assert_close(out.queryPositions, qp[p], rtol=0, atol=0)
    torch.testing.assert_close(out.referencePositions, rp[r], rtol=0, atol=0)
    torch.testing.assert_close(out.querySupports, qh[p], rtol=0, atol=0)
    torch.testing.assert_close(out.referenceSupports, rh[r], rtol=0, atol=0)
    assert adj.paddedJ.tolist() == [1,3,0,-1,-1]


@pytest.mark.parametrize('method', ['CompSPH', 'CRKSPH'])
@pytest.mark.parametrize('compatible', [False, True])
@pytest.mark.parametrize('adaptive', [False, True])
def test_sorted_rk_returns_input_order_and_matches_native_physics_and_canonical_export(method, compatible, adaptive, tmp_path):
    ctx, initial = shuffled(method, compatible=compatible, adaptive=adaptive)
    saved = initial.state.positions.clone()
    baseline = ctx.integrator(initial.initializeNewState(), 1e-4, ctx.stepFunction,
                              config=ctx.config, schemeConfig=ctx.schemeConfig)
    result = MortonStepper(ctx.integrator)(initial, 1e-4, config=ctx.config, schemeConfig=ctx.schemeConfig)
    assert result.step_status.successful and initial.spatialContext is None and initial.adjacency is None
    torch.testing.assert_close(initial.state.positions, saved, rtol=0, atol=0)
    assert result.state.adjacency is None and not result.state.spatialContext.layout.identity
    for name in ('positions','velocities','internalEnergies','supports','densities','UIDs'):
        torch.testing.assert_close(getattr(result.state.state, name), getattr(baseline.state.state, name), rtol=3e-6, atol=3e-7)
    for stage, control in zip(result.stages, baseline.stages):
        assert torch.equal(stage.aux[1].UIDs, initial.state.UIDs)
        spatial = stage.aux[2]
        assert particle_pairs(spatial.adjacency, spatial.layout.compute_ids) == particle_pairs(control.aux[0], initial.state.UIDs)
        for name in ('dxdt','dvdt','dudt'):
            torch.testing.assert_close(getattr(stage.update,name),getattr(control.update,name),rtol=3e-5,atol=3e-6)
    # Public export automatically makes a UID-ordered view, including stage CSR/anchors.
    from warpSPH.io.export import exportSimulationSystem
    exportSimulationSystem(str(tmp_path), 'frame', method, result.state, stages=result.stages, exportAdjacency=True)
    with h5py.File(tmp_path/'trajectory/frame.h5') as f:
        ids = f['state/UIDs'][:]
        assert ids.tolist() == sorted(initial.state.UIDs.tolist())
        for i, stage in enumerate(result.stages):
            group = f[f'stages/stage_{i}']
            assert group['aux/1_state/UIDs'][:].tolist() == ids.tolist()
            adjacency = group['aux/0_adjacency']
            pairs = sorted(zip(ids[adjacency['i'][:]].tolist(), ids[adjacency['j'][:]].tolist()))
            assert pairs == particle_pairs(stage.aux[2].adjacency, stage.aux[2].layout.compute_ids)
    accepted = adopt_spatial_layout(result)
    exportSimulationSystem(str(tmp_path), 'adopted', method, accepted.state, stages=accepted.stages, exportAdjacency=True)
    with h5py.File(tmp_path/'trajectory/adopted.h5') as f, h5py.File(tmp_path/'trajectory/frame.h5') as original:
        for name in ('positions','velocities','internalEnergies','supports','UIDs'):
            assert (f['state/'+name][:] == original['state/'+name][:]).all()
        for i in range(len(result.stages)):
            assert f[f'stages/stage_{i}/aux/1_state/UIDs'][:].tolist() == sorted(initial.state.UIDs.tolist())
    assert torch.equal(result.state.state.UIDs, initial.state.UIDs)


@pytest.mark.parametrize('method', ['CompSPH', 'CRKSPH'])
def test_adoption_reuses_list_and_next_step_identity_without_changing_old_snapshots(method, monkeypatch):
    ctx, initial = shuffled(method)
    calls = []
    stepper = MortonStepper(ctx.integrator, observe=lambda request, prepared:calls.append(prepared.adjacency))
    result = stepper(initial, 1e-4, config=ctx.config, schemeConfig=ctx.schemeConfig)
    old_ids = result.state.state.UIDs.clone()
    old_stage = result.stages[0].update.dvdt.clone()
    adopted = adopt_spatial_layout(result)
    context = result.state.spatialContext
    assert adopted.state.adjacency is context.adjacency and adopted.state.spatialContext.layout.identity
    assert torch.equal(adopted.state.state.UIDs, context.layout.compute_ids)
    assert torch.equal(result.state.state.UIDs, old_ids)
    assert torch.equal(result.stages[0].update.dvdt, old_stage)
    before = len(calls)
    import warpSPH.evaluation.sortedStep as boundary
    with monkeypatch.context() as patch:
        patch.setattr(boundary,'permute_state',lambda *a,**kw:pytest.fail('identity path remapped particle fields'))
        patch.setattr(boundary,'permute_update',lambda *a,**kw:pytest.fail('identity path remapped derivatives'))
        second = stepper(adopted.state, 1e-4, config=ctx.config, schemeConfig=ctx.schemeConfig)
    assert second.state.spatialContext.layout.identity
    assert all(adj is context.adjacency for adj in calls[before:])
    # Continuing without adoption follows the same physical trajectory through explicit maps.
    other = stepper(result.state, 1e-4, config=ctx.config, schemeConfig=ctx.schemeConfig)
    a, _ = canonical_view(second.state); b, _ = canonical_view(other.state)
    for name in ('positions','velocities','internalEnergies','supports','densities'):
        torch.testing.assert_close(getattr(a.state,name),getattr(b.state,name),rtol=0,atol=0)
    assert adopt_spatial_layout(adopted).state is adopted.state


def test_input_membership_changes_fail_before_using_same_size_cached_list():
    ctx, initial = shuffled()
    stepper = MortonStepper(ctx.integrator)
    result = stepper(initial, 1e-4, config=ctx.config, schemeConfig=ctx.schemeConfig)
    changed = result.state.initializeNewState()
    changed.state.UIDs[0] += 1000
    with pytest.raises(ValueError, match='membership/order changed'):
        stepper(changed, 1e-4, config=ctx.config, schemeConfig=ctx.schemeConfig)
    with pytest.raises(ValueError, match='history/reuse'):
        stepper(result.state, 1e-4, config=ctx.config, schemeConfig=ctx.schemeConfig, history=object())


@pytest.mark.parametrize('name', ['Backward Euler (implicit)', 'Gauss-Legendre 2'])
def test_sorted_fd_jfnk_keeps_fixed_vector_layout_and_returns_converged_input_order(name):
    ctx, initial = shuffled(compatible=False)
    initial.state.velocities += .05 * torch.sin(2*torch.pi*initial.state.positions)
    observed = []
    stepper = MortonStepper(get_scheme(name), observe=lambda request, prepared:observed.append(request.state.state.UIDs.clone()))
    with torch.no_grad():
        result = stepper(initial, 1e-4, config=ctx.config, schemeConfig=ctx.schemeConfig,
                         solver=JFNKSolver(tol=1e-10))
    assert result.step_status.successful
    ids = result.state.spatialContext.layout.compute_ids
    assert all(torch.equal(got,ids) for got in observed)
    tableau = get_scheme(name).definition.tableau
    for i, stage in enumerate(result.stages):
        for field, derivative in (('positions','dxdt'), ('velocities','dvdt'), ('internalEnergies','dudt')):
            expected = getattr(initial.state,field).clone()
            for j, other in enumerate(result.stages):expected += 1e-4*float(tableau.a[i][j])*getattr(other.update,derivative)
            torch.testing.assert_close(getattr(stage.aux[1],field),expected,rtol=1e-6,atol=1e-7)


@pytest.mark.parametrize('method', ['CompSPH', 'CRKSPH'])
def test_compatible_energy_replay_stays_in_one_compute_layout(method, monkeypatch):
    ctx, initial = shuffled(method)
    initial.state.velocities += .03 * torch.sin(2*torch.pi*initial.state.positions)
    control = MortonStepper(ctx.integrator)(initial, 1e-4, config=ctx.config, schemeConfig=ctx.schemeConfig)
    from warpSPH.modules import compressibleStage
    original = compressibleStage.stagePairList
    def changed(system, *args, **kwargs):
        adjacency = original(system, *args, **kwargs)
        return adjacency if system.pairWork.conservative else replace(adjacency)
    monkeypatch.setattr(compressibleStage, 'stagePairList', changed)
    seen = []
    result = MortonStepper(ctx.integrator, observe=lambda request, prepared:seen.append(request.state.state.UIDs.clone()))(
        initial, 1e-4, config=ctx.config, schemeConfig=ctx.schemeConfig)
    assert len(seen) == 8 and all(torch.equal(ids, seen[0]) for ids in seen)
    for name in ('positions','velocities','internalEnergies','supports','densities'):
        torch.testing.assert_close(getattr(result.state.state,name),getattr(control.state.state,name),rtol=3e-6,atol=3e-7)
    s0, s = initial.state, result.state.state
    def energy(s):return (s.masses.double()*(s.internalEnergies.double()+.5*s.velocities.double().square().sum(-1))).sum()
    torch.testing.assert_close(energy(s),energy(s0),rtol=2e-6,atol=1e-8)
    torch.testing.assert_close((s.masses[:,None]*s.velocities).double().sum(0),
                               (s0.masses[:,None]*s0.velocities).double().sum(0),rtol=1e-5,atol=1e-7)


def test_adoption_preserves_build_anchors_and_skin_usage():
    ctx, initial = shuffled(adaptive=False)
    result = MortonStepper(ctx.integrator)(initial,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig)
    spatial = result.state.spatialContext
    p = spatial.layout.p
    before = verletUsage(result.state.state.positions[p],result.state.state.supports[p],spatial.adjacency,ctx.config.domain,ctx.config.verletScale)
    saved = spatial.adjacency.queryPositions.clone()
    adopted = adopt_spatial_layout(result)
    after = verletUsage(adopted.state.state.positions,adopted.state.state.supports,adopted.state.adjacency,ctx.config.domain,ctx.config.verletScale)
    torch.testing.assert_close(before,after,rtol=0,atol=0)
    torch.testing.assert_close(adopted.state.adjacency.queryPositions,saved,rtol=0,atol=0)
    exported,_ = canonical_view(adopted.state)
    order = torch.argsort(spatial.layout.compute_ids)
    torch.testing.assert_close(exported.adjacency.queryPositions,saved[order],rtol=0,atol=0)


def test_stale_boundary_resort_composes_against_actual_input_and_leaves_old_context_intact():
    ctx, initial = shuffled(adaptive=False)
    stepper = MortonStepper(ctx.integrator)
    first = stepper(initial,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig)
    moved = first.state.initializeNewState()
    moved.state.positions = moved.state.positions.roll(7,0)
    old = first.state.spatialContext
    anchors = old.adjacency.queryPositions.clone()
    control = ctx.integrator(moved.initializeNewState(),1e-4,ctx.stepFunction,config=ctx.config,schemeConfig=ctx.schemeConfig)
    result = stepper(moved,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig)
    assert not torch.equal(result.state.spatialContext.layout.compute_ids,old.layout.compute_ids)
    assert torch.equal(result.state.spatialContext.layout.input_ids,moved.state.UIDs)
    for name in ('positions','velocities','internalEnergies','supports','densities'):
        torch.testing.assert_close(getattr(result.state.state,name),getattr(control.state.state,name),rtol=3e-6,atol=3e-7)
    torch.testing.assert_close(old.adjacency.queryPositions,anchors,rtol=0,atol=0)


def test_failed_jfnk_attempt_publishes_no_layout_or_scratch_and_fresh_retry_works():
    from warpSPHIntegrators.state import ResourceInvalidated, current_evaluation
    ctx, initial = shuffled(compatible=False)
    initial.state.velocities += .05*torch.sin(2*torch.pi*initial.state.positions)
    saved = initial.state.supports.clone()
    fail = [True]
    def observe(request, prepared):
        if fail[0] and request.purpose == 'fd_probe':raise ResourceInvalidated('test coverage renewal')
    stepper = MortonStepper(get_scheme('Backward Euler (implicit)'),observe=observe)
    with torch.no_grad(), pytest.raises(ResourceInvalidated,match='coverage renewal'):
        stepper(initial,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig,solver=JFNKSolver(tol=1e-10))
    assert initial.spatialContext is None and initial.adjacency is None and initial.pairWork is None
    torch.testing.assert_close(initial.state.supports,saved,rtol=0,atol=0)
    assert current_evaluation() is None
    fail[0] = False
    with torch.no_grad():
        result = stepper(initial,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig,solver=JFNKSolver(tol=1e-10))
    assert result.step_status.successful and result.state.spatialContext is not None


def test_embedded_error_is_mapped_with_adoption_and_failed_results_cannot_adopt():
    ctx, initial = shuffled(compatible=False)
    result = MortonStepper(get_scheme('Dormand-Prince 5(4)'))(initial,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig)
    assert result.error is not None
    p = result.state.spatialContext.layout.p
    accepted = adopt_spatial_layout(result)
    for name in ('positions','velocities','internalEnergies'):
        torch.testing.assert_close(getattr(accepted.error.state,name),getattr(result.error.state,name)[p],rtol=0,atol=0)
    bad = copy(result.state); bad.state = permute_state(result.state.state,torch.arange(p.numel()))
    bad.state.positions[0,0] = float('nan')
    with pytest.raises(ValueError,match='successful accepted'):
        adopt_spatial_layout(replace(result,state=bad))


def test_large_ids_globals_and_missing_field_axis_do_not_follow_tensor_length():
    ctx, initial = shuffled()
    s = initial.state
    s.UIDs = s.UIDs.long() + 2**40; s.UIDcounter = 2**40+32
    s.boundaryProjectionBudget = torch.arange(s.UIDs.numel(),dtype=s.positions.dtype)
    mapped = permute_state(s,torch.arange(s.UIDs.numel()-1,-1,-1))
    assert mapped.UIDcounter == s.UIDcounter
    torch.testing.assert_close(mapped.boundaryProjectionBudget,s.boundaryProjectionBudget,rtol=0,atol=0)
    result = MortonStepper(ctx.integrator)(initial,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig)
    assert torch.equal(result.state.state.UIDs,s.UIDs)
    s.unknownParticleData = torch.ones(s.UIDs.numel())
    with pytest.raises(TypeError,match='declare particle/global/edge axes'):
        permute_state(s,torch.arange(s.UIDs.numel()))


@pytest.mark.parametrize('dim,nx', [(2,8),(3,8)])
def test_multidimensional_native_pair_physics_matches_private_sorted_step(dim,nx):
    from warpSPH.configurations.simulationConfig import buildConfig
    from warpSPH.sample.compressible import setupBasicCompressibleInitialState
    from warpSPH.systems.compSPH import CompSPHState, CompSPHSystem
    from warpSPH.modules.eos import refreshEOS
    ctx, _ = setup('CompSPH',compatible=True)
    domain = buildDomainDescription(l=1,dim=dim,periodic=True,dtype=get_torch_precision())
    ctx.config, ctx.integrator = buildConfig(dim=dim,nx=nx,domain=domain,dtype=get_torch_precision(),
        device=torch.device('cpu'),dt=1e-4,integrationScheme=ctx.config.integrationScheme,n_h=3.)
    initial = setupBasicCompressibleInitialState(nx,ctx.config,ctx.schemeConfig,CompSPHState,CompSPHSystem)
    initial.state.internalEnergies.fill_(2.5)
    refreshEOS(initial.state,ctx.schemeConfig)
    initial.state.totalEnergies = initial.state.masses*initial.state.internalEnergies
    order = torch.randperm(initial.state.UIDs.numel(),generator=torch.Generator().manual_seed(21))
    initial.state = permute_state(initial.state,order); initial.adjacency = None
    initial.state.velocities[:,0] += .03*torch.sin(2*torch.pi*initial.state.positions[:,0])
    control = ctx.integrator(initial.initializeNewState(),1e-4,ctx.stepFunction,config=ctx.config,schemeConfig=ctx.schemeConfig)
    result = MortonStepper(ctx.integrator)(initial,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig)
    for name in ('positions','velocities','internalEnergies','supports','densities'):
        torch.testing.assert_close(getattr(result.state.state,name),getattr(control.state.state,name),rtol=5e-6,atol=5e-7)


def test_support_retry_rebuilds_in_fixed_compute_order_without_midstep_sort(monkeypatch):
    from warpSPH.modules import compressibleStage
    from warpSPHIntegrators.state import current_evaluation
    import warpSPH.evaluation.sortedStep as boundary
    ctx, initial = shuffled()
    original = compressibleStage.solveStageSupport
    calls, orders = [], []
    def grown(system, state, config, scheme, **kwargs):
        request = current_evaluation()
        if request.stage_path == (1,):
            state.supports = state.supports * (config.verletScale+2)
        original(system,state,config,scheme,**kwargs)
    monkeypatch.setattr(compressibleStage,'solveStageSupport',grown)
    sorter = boundary.morton_order
    def sort(*args,**kwargs):
        orders.append(True);return sorter(*args,**kwargs)
    monkeypatch.setattr(boundary,'morton_order',sort)
    def observe(request, prepared):calls.append((request.state.state.UIDs.clone(),prepared.adjacency))
    result = MortonStepper(ctx.integrator,observe=observe)(initial,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig)
    assert result.step_status.successful and len(orders) == 1
    assert all(torch.equal(ids,calls[0][0]) for ids,_ in calls)
    assert any(adj is not calls[0][1] for _,adj in calls)


def test_step_boundary_viscosity_memory_follows_adopted_particle_ids():
    from warpSPH.enumTypes import ViscositySwitch
    ctx, initial = shuffled()
    ctx.schemeConfig.viscositySwitchParams.scheme = ViscositySwitch.Rosswog2020
    initial.state.velocities += .02*torch.sin(2*torch.pi*initial.state.positions)
    stepper = MortonStepper(ctx.integrator)
    unadopted = initial
    adopted = initial
    for _ in range(3):
        unadopted = stepper(unadopted,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig).state
        adopted = adopt_spatial_layout(stepper(adopted,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig)).state
    a,_ = canonical_view(unadopted);b,_ = canonical_view(adopted)
    for name in ('positions','internalEnergies','alphas','entropiesPrev','entropyRates'):
        torch.testing.assert_close(getattr(a.state,name),getattr(b.state,name),rtol=0,atol=0)
    assert a.state.entropiesPrevTime == b.state.entropiesPrevTime


def test_empty_csr_export_and_aliased_self_anchors():
    for n in (0,3):
        counts = torch.zeros(n,dtype=torch.int32); offsets = torch.zeros(n,dtype=torch.int64)
        anchors = torch.arange(float(n)).reshape(n,1);h = torch.ones(n)
        adj = AdjacencyList(None,torch.full((2,),-1,dtype=torch.int32),counts,offsets,n,n,
                            anchors,anchors,h,h,numEdges=0)
        result = remap_adjacency(adj,torch.arange(n-1,-1,-1))
        assert result.queryPositions is result.referencePositions
        assert result.querySupports is result.referenceSupports
        assert result.paddedJ.tolist() == [-1,-1] and result.numEdges == 0


@pytest.mark.parametrize('method', ['CompSPH','CRKSPH'])
def test_explicit_reverse_gradient_is_in_the_original_particle_basis(method):
    ctx, initial = shuffled(method,compatible=True,adaptive=False)
    initial.state.velocities = (.03*torch.sin(2*torch.pi*initial.state.positions)).detach().requires_grad_()
    parameter = initial.state.velocities
    baseline = ctx.integrator(initial.initializeNewState(),1e-4,ctx.stepFunction,config=ctx.config,schemeConfig=ctx.schemeConfig)
    result = MortonStepper(ctx.integrator)(initial,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig)
    def loss(system):
        weights = system.state.UIDs.to(parameter.dtype)+1
        return (weights*(system.state.velocities.square().sum(-1)+.01*system.state.internalEnergies)).sum()
    native = torch.autograd.grad(loss(baseline.state),parameter)[0]
    actual = torch.autograd.grad(loss(result.state),parameter)[0]
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual,native,rtol=5e-5,atol=5e-6)


@pytest.mark.parametrize('bad', ['source','boundary','inplace','history','preconditioner','tolerance','partial_step','unknown_field'])
def test_unadapted_inputs_fail_before_sort_or_equations(bad,monkeypatch):
    import warpSPH.evaluation.sortedStep as boundary
    ctx, initial = shuffled(compatible=False)
    stepper = MortonStepper(get_scheme('Backward Euler (implicit)'))
    kwargs = {}
    if bad == 'source':initial.sourceTerms = object()
    elif bad == 'boundary':ctx.schemeConfig.boundaryConditions = [object()]
    elif bad == 'inplace':ctx.schemeConfig.inPlaceAdjacency = True
    elif bad == 'history':kwargs['history'] = object()
    elif bad == 'preconditioner':kwargs['solver'] = JFNKSolver(preconditioner=lambda *a: a[0])
    elif bad == 'tolerance':kwargs['atol'] = torch.ones(initial.state.UIDs.numel())
    elif bad == 'partial_step':kwargs['_finalize'] = 0
    else:initial.state.newField = torch.ones(initial.state.UIDs.numel())
    monkeypatch.setattr(boundary,'morton_order',lambda *a,**kw:pytest.fail('invalid input reached sorting'))
    with pytest.raises((ValueError,TypeError)):
        stepper(initial,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig,**kwargs)
    assert initial.spatialContext is None


def test_plain_driver_can_leave_private_cache_and_export_stays_uid_ordered(tmp_path):
    from warpSPH.io.export import exportSimulationSystem
    ctx, initial = shuffled()
    first = MortonStepper(ctx.integrator)(initial,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig)
    accepted = adopt_spatial_layout(first).state
    # The plain driver uses the already-adopted list in its own input basis.
    plain = ctx.integrator(accepted.initializeNewState(),1e-4,ctx.stepFunction,config=ctx.config,schemeConfig=ctx.schemeConfig)
    assert plain.state.spatialContext is None and plain.state.canonicalExport
    exportSimulationSystem(str(tmp_path),'plain','CompSPH',plain.state,stages=plain.stages)
    with h5py.File(tmp_path/'trajectory/plain.h5') as f:
        ids = f['state/UIDs'][:]
        assert ids.tolist() == sorted(ids.tolist())
        for i in range(len(plain.stages)):
            assert f[f'stages/stage_{i}/aux/1_state/UIDs'][:].tolist() == ids.tolist()


def test_unknown_axis_is_also_rejected_on_the_adopted_identity_path():
    ctx, initial = shuffled()
    stepper = MortonStepper(ctx.integrator)
    accepted = adopt_spatial_layout(stepper(initial,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig)).state
    accepted.state.unknownAxis = torch.ones(accepted.state.UIDs.numel())
    with pytest.raises(TypeError,match='declare particle/global/edge axes'):
        stepper(accepted,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig)


def test_unlabelled_export_extras_and_fixed_roster_writer_cannot_silently_mix_orders(tmp_path):
    from warpSPH.io.export import exportSimulationSystem, writeFrame, writeInitialData
    ctx, initial = shuffled()
    result = MortonStepper(ctx.integrator)(initial,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig)
    with pytest.raises(ValueError,match='extraData needs explicit axes'):
        exportSimulationSystem(str(tmp_path),'bad','CompSPH',result.state,extraData={'rowData':torch.arange(32)})
    assert not (tmp_path/'trajectory').exists()
    with pytest.raises(ValueError,match='ID/schema adapter'):
        writeFrame(None,0,result.state,result.stages,ctx.config,ctx.schemeConfig)
    with pytest.raises(ValueError,match='ID/schema adapter'):
        writeInitialData(None,None,'CompSPH',ctx.config,ctx.schemeConfig,None,result.state)


def test_particle_only_export_does_not_remap_edges_and_pair_fields_keep_their_edge_owner(monkeypatch,tmp_path):
    import warpSPH.evaluation.sortedStep as boundary
    from warpSPH.io.export import exportSimulationSystem
    ctx, initial = shuffled(compatible=False)
    result = MortonStepper(ctx.integrator)(initial,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig)
    view, stages = canonical_view(result.state,result.stages)
    # Exact physical-pair keys, independent of row and edge permutations.
    for original, exported in zip(result.stages, stages):
        spatial = original.aux[2]
        i,j = spatial.adjacency.livePairs(); ni,nj = exported.aux[0].livePairs()
        for name, values in spatial.edge_fields.items():
            if values is None:continue
            old = {(int(spatial.layout.compute_ids[a]),int(spatial.layout.compute_ids[b])):v
                   for a,b,v in zip(i,j,values[:i.numel()])}
            data = getattr(exported.aux[1],name)
            for a,b,v in zip(ni,nj,data[:ni.numel()]):
                torch.testing.assert_close(v,old[(int(exported.aux[1].UIDs[a]),int(exported.aux[1].UIDs[b]))],rtol=0,atol=0)
    with monkeypatch.context() as patch:
        patch.setattr(boundary,'remap_adjacency',lambda *a,**kw:pytest.fail('particle-only export allocated an edge remap'))
        exportSimulationSystem(str(tmp_path),'particles','CompSPH',result.state,
                               stages=result.stages,exportAdjacency=False,exportStagesAdjacency=False)
    with h5py.File(tmp_path/'trajectory/particles.h5') as f:
        assert 'adjacency' not in f
        assert '0_adjacency' not in f['stages/stage_0/aux']


def test_every_declared_particle_field_follows_physical_ids_not_its_length():
    from warpSPH.evaluation.layout import PARTICLE_FIELDS, EDGE_FIELDS
    _, initial = shuffled()
    state = initial.state
    n = state.UIDs.numel()
    for number, name in enumerate(sorted(PARTICLE_FIELDS)):
        values = torch.arange(n,dtype=state.positions.dtype) + 100*number
        if name in ('positions','velocities','wallNormals'):values = torch.stack((values,values+.5),-1)
        if name in ('kinds','materials','emitters','bodyIDs'):values = values.long()
        setattr(state,name,values)
    state.UIDs = torch.arange(n,dtype=torch.int64) + 2**40
    state.UIDcounter = 2**40+n
    state.boundaryProjectionBudget = torch.arange(n,dtype=state.positions.dtype)
    for name in EDGE_FIELDS:setattr(state,name,torch.arange(n,dtype=state.positions.dtype))
    # Independent record join: every column belongs to the particle's physical ID.
    records = {int(state.UIDs[row]):{name:getattr(state,name)[row].tolist() for name in PARTICLE_FIELDS}
               for row in range(n)}
    result = permute_state(state,torch.randperm(n,generator=torch.Generator().manual_seed(83)))
    for row, uid in enumerate(result.UIDs):
        assert {name:getattr(result,name)[row].tolist() for name in PARTICLE_FIELDS} == records[int(uid)]
    assert result.UIDcounter == state.UIDcounter
    torch.testing.assert_close(result.boundaryProjectionBudget,state.boundaryProjectionBudget,rtol=0,atol=0)
    assert all(getattr(result,name) is None for name in EDGE_FIELDS)
    assert all(getattr(state,name) is not None for name in EDGE_FIELDS)
