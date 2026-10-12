"""Geometry-only probe (no time stepping): on saved KH 256 MFM states, how often does
the closure correction A' = A_g + |A_g| (lam_j - lam_i) flip a face against its pair
axis (A'.d < 0, which the guards reject for A_g but never re-check after the
projection), and how large does it make faces relative to the guarded A_g?"""
import os, sys
os.environ['warpSPHCore_PRECISION'] = 'float32'
sys.path.insert(0, '/home/lu26029/dev/warpSPH/src')
import torch, warp as wp
torch.set_grad_enabled(False)
wp.init()
from warpSPHCore.enumTypes import KernelFunctions
from warpSPHCore.dataTypes import DomainDescription, ParticleState
from warpSPHCore.mfm import MeshlessWarp
from warpSPHCore.mfm.wp_mfm import guardedFace

@wp.kernel
def faceK(d: wp.array(dtype=wp.vec2f), i: wp.array(dtype=wp.int32), j: wp.array(dtype=wp.int32), V: wp.array(dtype=wp.float32),
          E: wp.array(dtype=wp.mat22f), h: wp.array(dtype=wp.float32), bad: wp.array(dtype=wp.int32), kernel: wp.int32,
          out: wp.array(dtype=wp.vec2f)):
    t = wp.tid()
    a = i[t]
    b = j[t]
    out[t] = guardedFace(d[t], V[a], V[b], E[a], E[b], h[a], h[b], bad[a], bad[b], kernel, 2, 1, 0)


dev = 'cuda'
import time
domain = DomainDescription(dim=2, min=torch.zeros(2, device=dev), max=torch.ones(2, device=dev),
                           periodic=torch.ones(2, dtype=torch.bool, device=dev))
kernel = KernelFunctions.B7
T = '/home/lu26029/dev/warpSPH/.tmp/'
CAPS = [0.0, 1.0, 0.75, 0.5, 0.25]
for name in ['kh256_state_step001000_periodic.pt', 'kh256_state_step003000_periodic.pt', 'kh256_state_step003379_warn500.pt',
             'kh256_state_step005000_periodic.pt', 'kh256_state_step006000_window.pt']:
    dd = torch.load(T + name, map_location='cpu', weights_only=False)
    pos, h = dd['positions'].to(dev).contiguous(), dd['supports'].to(dev).contiguous()
    N = pos.shape[0]
    print(f'== {name} N={N}', flush=True)
    for cap in CAPS:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        gw = MeshlessWarp.build(pos, h, domain, kernel, cond_max=1e3, closure='project', closure_power=1.0,
                                cg_iters=200, guards=True, centred_weights=True, area_cap=False,
                                closure_solver='auto', mg_min_particles=2048, mg_max_coarse=512, closure_cap=cap)
        torch.cuda.synchronize(); dt_ms = 1e3 * (time.perf_counter() - t0)
        lam = gw.lam
        adj = gw.adjacency
        i = adj.i.to(torch.int32); j = adj.j.to(torch.int32)
        keep = i < j
        i, j = i[keep].contiguous(), j[keep].contiguous()
        d = pos[j] - pos[i]
        d = d - torch.round(d)
        r = d.norm(dim=-1)
        inside = r <= torch.maximum(h[i], h[j])
        i, j, d = i[inside].contiguous(), j[inside].contiguous(), d[inside].contiguous()
        Ag = torch.zeros_like(d)
        wp.launch(faceK, dim=i.numel(), inputs=[wp.from_torch(d, dtype=wp.vec2f), wp.from_torch(i), wp.from_torch(j),
                  wp.from_torch(gw.volume.contiguous()), wp.from_torch(gw.Einv.contiguous(), dtype=wp.mat22f),
                  wp.from_torch(h), wp.from_torch(gw.condBad.to(torch.int32).contiguous()), int(kernel.value)],
                  outputs=[wp.from_torch(Ag, dtype=wp.vec2f)])
        mag = Ag.norm(dim=-1)
        dlam = lam[j] - lam[i]
        if cap > 0:
            m = dlam.norm(dim=-1, keepdim=True)
            dlam = dlam * torch.clamp(cap / m.clamp_min(1e-30), max=1.0)
        Ap = Ag + mag[:, None] * dlam
        flip = ((Ap * d).sum(-1) < 0) & ((Ag * d).sum(-1) >= 0)
        ratio = Ap.norm(dim=-1) / mag.clamp_min(1e-30)
        def resid(A):
            R = torch.zeros(N, 2, device=dev).index_add_(0, i.long(), A).index_add_(0, j.long(), -A)
            return R.norm()
        raw = resid(Ag); res = resid(Ap)
        q = lambda x, p: torch.quantile(x.float()[:2_000_000], p).item()
        print(f'  cap {cap:4.2f}: flipped {flip.float().mean()*100:6.3f}% ({int(flip.sum()):6d}) | |A\'|/|A_g| p99.9 {q(ratio,0.999):6.2f} max {ratio.max():7.2f} | '
              f'true residual / raw {float(res/raw):.3f} | outer {gw.closureOuterIterations} sat {float(gw.closureSaturated)*100:5.2f}% '
              f'cgIters {int(gw.closureIterations)} | build {dt_ms:6.1f} ms', flush=True)
