"""Analytical thermodynamics and configuration contracts for pure materials."""
from types import SimpleNamespace

import pytest
import torch

from warpSPH.configurations import (CRKSPHConfig, EOSConfig, MaterialConfig,
    crkSPHConfigToDict, dictToCRKSPHConfig)
from warpSPH.modules.eos import evaluateEOS, validateMaterialState
from warpSPH.modules.eos.materials import materialRuntime
from warpSPH.modules.eos.idealGas import idealGasEOS
from warpSPH.runner import RunConfiguration
from warpSPHCore.type_config import get_torch_precision


def config():
    return CRKSPHConfig(materials=[MaterialConfig('air', EOSConfig(gamma=1.4)),
                                  MaterialConfig('gas', EOSConfig(gamma=5/3))])


@pytest.mark.parametrize('source', ['u', 'P', 'A', 'c'])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required'))])
def test_analytic_roundtrips_and_permutation(source, device):
    cfg = config()
    dtype = get_torch_precision()
    ids = torch.tensor([1, 0, 1, 0], dtype=torch.int32, device=device)
    rho = torch.tensor([3., .7, 1.2, 2.], dtype=dtype, device=device)
    gamma = torch.tensor([5/3, 1.4, 5/3, 1.4], dtype=dtype, device=device)
    u = torch.tensor([2., 3., 4., 1.], dtype=dtype, device=device)
    P = (gamma - 1) * rho * u
    A = P / rho**gamma
    c = (gamma * P / rho).sqrt()
    result = evaluateEOS(rho, ids, cfg, **{source: {'u': u, 'P': P, 'A': A, 'c': c}[source]})
    for actual, expected in zip(result, (A, u, P, c)):
        torch.testing.assert_close(actual, expected)
    assert materialRuntime(cfg, rho) is materialRuntime(cfg, rho)


def test_uniform_fallback_ignores_region_labels():
    rho, u = torch.ones(4), torch.arange(1., 5.)
    cfg = CRKSPHConfig()
    actual = evaluateEOS(rho, torch.tensor([0, 7, -2, 42], dtype=torch.int32), cfg, u=u)
    expected = idealGasEOS(A=None, u=u, P=None, rho=rho, gamma=cfg.gamma)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_device_guard_and_negative_energy_are_visible():
    cfg = config()
    rho = torch.ones(4)
    result = evaluateEOS(rho, torch.tensor([-1, 2, 0, 1], dtype=torch.int32), cfg,
                         u=torch.tensor([1., 1., -1., 1.]))
    assert all(torch.isnan(x[:2]).all() for x in result)
    assert torch.isnan(result[3][2])
    assert result[2][2] < 0
    assert torch.isfinite(result[3][3])


def test_empty_and_bad_source():
    cfg = config()
    empty = torch.empty(0)
    assert all(x.numel() == 0 for x in evaluateEOS(empty, empty.int(), cfg, u=empty))
    with pytest.raises(ValueError, match='exactly one'):
        evaluateEOS(empty, empty.int(), cfg, u=empty, P=empty)
    cfg.materials = []
    with pytest.raises(ValueError, match='nonempty'):
        evaluateEOS(empty, empty.int(), cfg, u=empty)


@pytest.mark.parametrize('ids', [torch.tensor([-1, 0], dtype=torch.int32),
                                torch.tensor([0, 2], dtype=torch.int32),
                                torch.tensor([0., 1.])])
def test_invalid_material_state(ids):
    state = SimpleNamespace(materials=ids, densities=torch.ones(2), kinds=torch.zeros(2, dtype=torch.int32))
    with pytest.raises(ValueError, match='materials'):
        validateMaterialState(state, config())


def test_parameter_roundtrip_and_runtime_invalidation():
    cfg = config()
    restored = dictToCRKSPHConfig(crkSPHConfigToDict(cfg))
    assert restored.materials == cfg.materials
    rho = torch.ones(1)
    original = materialRuntime(cfg, rho)
    cfg.materials[0] = MaterialConfig('changed', EOSConfig(gamma=1.5))
    assert materialRuntime(cfg, rho) is not original
    old = crkSPHConfigToDict(CRKSPHConfig())
    old.pop('materials')
    assert dictToCRKSPHConfig(old).materials is None


def test_gradients_and_forward_tangent():
    cfg = config()
    ids = torch.tensor([1, 0], dtype=torch.int32)
    rho = torch.tensor([2., 3.], dtype=torch.float64, requires_grad=True)
    u = torch.tensor([3., 4.], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda r, e: evaluateEOS(r, ids, cfg, u=e), (rho, u))
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(u.detach(), torch.ones_like(u))
        _, _, P, _ = evaluateEOS(rho.detach(), ids, cfg, u=dual)
        tangent = torch.autograd.forward_ad.unpack_dual(P).tangent
        torch.testing.assert_close(tangent, rho.detach() * torch.tensor([2/3, .4], dtype=rho.dtype))


def test_configuration_schema_and_unsupported_switch():
    from warpSPH.cases.compressible.sod import sodCase
    from warpSPH.configurations.moduleConfigurations.materials import materialsToDict
    document = RunConfiguration('sod', solver={'scheme': 'CRKSPH', 'materials': materialsToDict(config().materials)})
    ctx = document.prepare(sodCase)
    assert ctx.schemeConfig.materials == config().materials
    document.solver['viscositySwitchParams'] = {'scheme': 'ReadHayfield2012'}
    with pytest.raises(ValueError, match='materials'):
        document.prepare(sodCase)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_eos_graph_replays_current_source():
    cfg = config()
    rho = torch.tensor([1., 3.], device='cuda')
    ids = torch.tensor([0, 1], device='cuda', dtype=torch.int32)
    u = torch.ones_like(rho)
    evaluateEOS(rho, ids, cfg, u=u)  # compile/upload outside capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = evaluateEOS(rho, ids, cfg, u=u)
    u.fill_(2.)
    graph.replay()
    torch.testing.assert_close(actual[2], torch.tensor([.8, 4.], device='cuda'))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_table_creation_and_evaluation_on_torch_side_stream():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        rho = torch.tensor([1., 3.], device='cuda', dtype=get_torch_precision())
        ids = torch.tensor([0, 1], device='cuda', dtype=torch.int32)
        result = evaluateEOS(rho, ids, config(), u=torch.full_like(rho, 2.))
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(result[2], torch.tensor([.8, 4.], device='cuda', dtype=rho.dtype))


def test_richtmyer_schema_and_reference_states():
    import json
    from warpSPH.cases.compressible.richtmyerMeshkov import richtmyerMeshkovCase
    from warpSPH.caseUtils.compressible.richtmyerMeshkov import incidentShock, interfaceStarState
    from warpSPH.runner import configurationSchema
    json.dumps(configurationSchema(richtmyerMeshkovCase), allow_nan=False)
    assert RunConfiguration('richtmyerMeshkov', variant='smoke').resolve(richtmyerMeshkovCase).nx == 16
    shock = incidentShock(1., 1., 1.4, 1.2)
    assert shock['density'] == pytest.approx(1.341614906832298)
    assert shock['pressure'] == pytest.approx(1.5133333333333332)
    star = interfaceStarState((1., 0., 1., 1.4), (3., 0., 1., 5/3))
    assert star['velocity'] == pytest.approx(0, abs=1e-14)
    assert star['pressure'] == pytest.approx(1)
    # Classic Sod shock tube: the independent root agrees with the known star state.
    sod = interfaceStarState((1., 0., 1., 1.4), (.125, 0., .1, 1.4))
    assert sod['pressure'] == pytest.approx(.303130178, rel=1e-7)
    assert sod['velocity'] == pytest.approx(.92745262, rel=1e-7)
