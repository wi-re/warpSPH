"""warp.optim.linear CG on the REAL IISPH pressure operator (TGV, divergenceFree state): eager host-checked, conditional graph, nested in a torch CUDA graph.
    python scripts/probe_warpOptimLinearReal.py 80 eager|graph|nested     # nx, mode
(see docs/linalg/SOLVER_INVENTORY.md section 8)
"""
import sys, time, traceback
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
import torch, warp as wp
import warp.optim.linear as L
from warpSPH.cases import importAll
from warpSPH.runner import run
from warpSPH.runner.case import getCase
from warpSPH.modules.incompressible.krylov import buildIISPHMatvec, buildIISPHPrecond
from warpSPH.modules.shifting.cg import cgSolve

importAll()
nx = int(sys.argv[1]) if len(sys.argv) > 1 else 80
r = run(getCase('tgv'), quiet=True, progress=False, nSteps=5, scheme='divergenceFree', plot=False, video=False, nx=nx)
system, ctx = r.state, r.ctx
state, config, schemeConfig, adjacency = system.state, ctx.config, ctx.schemeConfig, system.adjacency
dt = float(config.dt)
n = state.positions.shape[0]
dev = state.positions.device
wdev = wp.get_device('cuda:0')
print(f'N = {n}')

raw = buildIISPHMatvec(state, config, schemeConfig, adjacency, dt)           # A p (negative definite)
precond = buildIISPHPrecond(state, config, schemeConfig, adjacency, dt)      # 1/D, D < 0
torch.manual_seed(1)
b_t = -(torch.randn(n, device=dev) * 1e-3)
b_t = b_t - b_t.mean()

# positive-definite system for CG: (-A) p = -b
negA = lambda p: -raw(p)
rhs = negA(torch.randn(n, device=dev)); rhs = rhs - rhs.mean()
prec_pos = -precond


def tmatvec(x, y, z, alpha, beta):
    xt, yt, zt = wp.to_torch(x), wp.to_torch(y), wp.to_torch(z)
    cs = wp.get_stream(wdev).cuda_stream
    with torch.cuda.stream(torch.cuda.ExternalStream(cs)):
        zt.copy_(float(alpha) * negA(xt) + float(beta) * yt)


@wp.kernel
def mul_kernel(x: wp.array(dtype=wp.float32), y: wp.array(dtype=wp.float32), z: wp.array(dtype=wp.float32),
               alpha: wp.float32, beta: wp.float32, s: wp.array(dtype=wp.float32)):
    t = wp.tid()
    z[t] = alpha * s[t] * x[t] + beta * y[t]


s_w = wp.from_torch(prec_pos.contiguous())
M = L.LinearOperator((n, n), wp.float32, wdev,
                     lambda x, y, z, alpha, beta: wp.launch(mul_kernel, dim=n, inputs=[x, y, z, float(alpha), float(beta), s_w], device=wdev))
A = L.LinearOperator((n, n), wp.float32, wdev, tmatvec)
rtol = 1e-4


def rel(x_t):
    return float((negA(x_t) - rhs).norm() / rhs.norm())


# reference
cgSolve(negA, rhs, torch.zeros(n, device=dev), rtol=rtol, maxiter=5, precond=prec_pos)
torch.cuda.synchronize()
t0 = time.perf_counter()
xx, st, hist = cgSolve(negA, rhs, torch.zeros(n, device=dev), rtol=rtol, maxiter=400, precond=prec_pos)
torch.cuda.synchronize()
print(f'{"shifting.cgSolve":<44} {(time.perf_counter() - t0) * 1e3:8.1f} ms  status {st:4d}  rel residual {rel(xx):.2e}')


def go(label, **kw):
    b = wp.from_torch(rhs.clone())
    try:
        L.cg(A, b, wp.zeros(n, dtype=wp.float32, device=wdev), tol=rtol, maxiter=5, M=M, **kw)
        torch.cuda.synchronize()
        x = wp.zeros(n, dtype=wp.float32, device=wdev)
        t0 = time.perf_counter()
        out = L.cg(A, b, x, tol=rtol, maxiter=400, M=M, **kw)
        torch.cuda.synchronize()
        it = out[0]
        it = int(it.numpy()[0]) if hasattr(it, 'numpy') else it
        print(f'{label:<44} {(time.perf_counter() - t0) * 1e3:8.1f} ms  iters {it:4d}  rel residual {rel(wp.to_torch(x)):.2e}', flush=True)
    except Exception as e:                                                                  # noqa: BLE001
        print(f'{label:<44} FAIL {type(e).__name__}: {str(e).splitlines()[0][:160]}', flush=True)


mode = sys.argv[2] if len(sys.argv) > 2 else 'eager'
if mode == 'eager':
    go('warp cg, host check/10, no graph', check_every=10, use_cuda_graph=False)
elif mode == 'graph':
    go('warp cg, check_every=0, conditional graph', check_every=0, use_cuda_graph=True)
elif mode == 'nested':
    try:
        b = wp.from_torch(rhs.clone())
        L.cg(A, b, wp.zeros(n, dtype=wp.float32, device=wdev), tol=rtol, maxiter=5, M=M, check_every=10, use_cuda_graph=False)
        torch.cuda.synchronize()
        x = wp.zeros(n, dtype=wp.float32, device=wdev)
        stream = torch.cuda.Stream()
        wstream = wp.stream_from_torch(stream)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), wp.ScopedStream(wstream, sync_enter=False, sync_exit=False), torch.cuda.graph(g, stream=stream):
            wp.capture_begin(device=wdev, stream=wstream, external=True)
            L.cg(A, b, x, tol=rtol, maxiter=400, M=M, check_every=0, use_cuda_graph=True)
        x.zero_()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        print(f'{"warp cg nested in torch graph (replay)":<44} {(time.perf_counter() - t0) * 1e3:8.1f} ms  rel residual {rel(wp.to_torch(x)):.2e}')
    except Exception as e:                                                                  # noqa: BLE001
        print(f'nested FAIL {type(e).__name__}: {str(e).splitlines()[0][:200]}')
        traceback.print_exc(limit=4)
