"""Can warp.optim.linear drive our operators?  n = 256^2 synthetic Laplacian: pure-warp matvec vs torch-closure matvec,
host-checked vs conditional-graph (check_every=0) vs inside a torch CUDA graph.  Prints solve time, iterations, true residual.
    ONLY=torch-eager GRAPH=1 python scripts/probe_warpOptimLinear.py     # one scenario per process: warp-eager | torch-eager | warp-nested | torch-nested
    (a failed capture can poison the CUDA context, so scenarios are run separately; see docs/linalg/SOLVER_INVENTORY.md section 8)
"""
import sys, time, traceback
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
import torch, warp as wp
import warp.optim.linear as L

dev = torch.device('cuda')
wdev = wp.get_device('cuda:0')
m = 256
n = m * m
shift = 1e-2


# ---- operators -------------------------------------------------------------------------------------------------------------------
@wp.kernel
def lap_kernel(x: wp.array(dtype=wp.float32), y: wp.array(dtype=wp.float32), z: wp.array(dtype=wp.float32),
               alpha: wp.float32, beta: wp.float32, m: int, shift: wp.float32):
    t = wp.tid()
    i = t // m
    j = t % m
    c = x[t]
    acc = (4.0 + shift) * c
    if i > 0:
        acc -= x[t - m]
    if i < m - 1:
        acc -= x[t + m]
    if j > 0:
        acc -= x[t - 1]
    if j < m - 1:
        acc -= x[t + 1]
    z[t] = alpha * acc + beta * y[t]


def warp_matvec(x, y, z, alpha, beta):
    wp.launch(lap_kernel, dim=n, inputs=[x, y, z, float(alpha), float(beta), m, shift], device=wdev)


def torch_A_correct(xt):
    g = xt.view(m, m)
    out = (4.0 + shift) * g
    out[1:, :] -= g[:-1, :]
    out[:-1, :] -= g[1:, :]
    out[:, 1:] -= g[:, :-1]
    out[:, :-1] -= g[:, 1:]
    return out.reshape(-1)


def torch_matvec(x, y, z, alpha, beta):
    xt, yt, zt = wp.to_torch(x), wp.to_torch(y), wp.to_torch(z)
    cs = wp.get_stream(wdev).cuda_stream
    with torch.cuda.stream(torch.cuda.ExternalStream(cs)):
        zt.copy_(float(alpha) * torch_A_correct(xt) + float(beta) * yt)


torch.manual_seed(0)
xs = torch.randn(n, device=dev)
b_t = torch_A_correct(xs)
inv_t = torch.full((n,), 1.0 / (4.0 + shift), device=dev)


def make_ops(kind):
    if kind == 'warp':
        A = L.LinearOperator((n, n), wp.float32, wdev, warp_matvec)
    else:
        A = L.LinearOperator((n, n), wp.float32, wdev, torch_matvec)
    M = L.preconditioner(A, 'diag') if False else L.LinearOperator((n, n), wp.float32, wdev,
        lambda x, y, z, alpha, beta: wp.launch(scale_kernel, dim=n, inputs=[x, y, z, float(alpha), float(beta), 1.0 / (4.0 + shift)], device=wdev))
    return A, M


@wp.kernel
def scale_kernel(x: wp.array(dtype=wp.float32), y: wp.array(dtype=wp.float32), z: wp.array(dtype=wp.float32),
                 alpha: wp.float32, beta: wp.float32, s: wp.float32):
    t = wp.tid()
    z[t] = alpha * s * x[t] + beta * y[t]


def resid(xw):
    xt = wp.to_torch(xw)
    return float((torch_A_correct(xt) - b_t).norm() / b_t.norm())


def run(label, kind, **kw):
    A, M = make_ops(kind)
    b = wp.from_torch(b_t.clone())
    x = wp.zeros(n, dtype=wp.float32, device=wdev)
    try:
        # warm up (module load) on a throwaway solve
        L.cg(A, b, wp.zeros(n, dtype=wp.float32, device=wdev), tol=1e-4, maxiter=5, M=M, **kw)
        torch.cuda.synchronize()
        x = wp.zeros(n, dtype=wp.float32, device=wdev)
        t0 = time.perf_counter()
        out = L.cg(A, b, x, tol=1e-4, maxiter=400, M=M, **kw)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1e3
        it = out[0]
        it = int(it.numpy()[0]) if hasattr(it, 'numpy') else it
        print(f'{label:<46} OK   {dt:8.1f} ms   iters {it:4d}   rel residual {resid(x):.2e}', flush=True)
    except Exception as e:                                                                  # noqa: BLE001
        print(f'{label:<46} FAIL {type(e).__name__}: {str(e).splitlines()[0][:150]}', flush=True)


import os
ONLY = os.environ.get('ONLY')
for kind in ('warp', 'torch'):
    if ONLY and ONLY != f'{kind}-eager':
        continue
    run(f'{kind} matvec, host check every 10, no graph', kind, check_every=10, use_cuda_graph=False)
    for lab, kw in (('host check every 10, graph', dict(check_every=10, use_cuda_graph=True)), ('check_every=0, conditional graph', dict(check_every=0, use_cuda_graph=True))):
        if os.environ.get('GRAPH'):
            run(f'{kind} matvec, {lab}', kind, **kw)

# ---- inside a torch CUDA graph (how the repo captures a step) ----------------------------------------------------------------------
print('--- nested in torch.cuda.graph')
for kind in ('warp', 'torch'):
    if ONLY != f'{kind}-nested':
        continue
    try:
        A, M = make_ops(kind)
        b = wp.from_torch(b_t.clone())
        x = wp.zeros(n, dtype=wp.float32, device=wdev)
        L.cg(A, b, wp.zeros(n, dtype=wp.float32, device=wdev), tol=1e-4, maxiter=5, M=M, check_every=10, use_cuda_graph=False)
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        wstream = wp.stream_from_torch(stream)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), wp.ScopedStream(wstream, sync_enter=False, sync_exit=False), torch.cuda.graph(g, stream=stream):
            wp.capture_begin(device=wdev, stream=wstream, external=True)
            try:
                L.cg(A, b, x, tol=1e-4, maxiter=400, M=M, check_every=0, use_cuda_graph=True)
            finally:
                pass
        x.zero_()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        print(f'{kind:<46} OK   {(time.perf_counter() - t0) * 1e3:8.1f} ms (replay)   rel residual {resid(x):.2e}', flush=True)
    except Exception as e:                                                                  # noqa: BLE001
        print(f'{kind:<46} FAIL {type(e).__name__}: {str(e).splitlines()[0][:200]}', flush=True)
        traceback.print_exc(limit=3)

# ---- reference: our torch cgSolve on the same system -------------------------------------------------------------------------------
from warpSPH.modules.shifting.cg import cgSolve
cgSolve(torch_A_correct, b_t, torch.zeros(n, device=dev), rtol=1e-4, maxiter=5, precond=inv_t)
torch.cuda.synchronize()
t0 = time.perf_counter()
xx, st, hist = cgSolve(torch_A_correct, b_t, torch.zeros(n, device=dev), rtol=1e-4, maxiter=400, precond=inv_t)
torch.cuda.synchronize()
print(f'{"reference: shifting.cgSolve (torch)":<46} OK   {(time.perf_counter() - t0) * 1e3:8.1f} ms   iters {st:4d}   rel residual {float((torch_A_correct(xx) - b_t).norm() / b_t.norm()):.2e}')
