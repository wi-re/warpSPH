#!/usr/bin/env python3
"""Build read-only every-step post-shift probes into an isolated dynGPUSPH copy.

DYNGPU_ACCEPTED_PROBES enables the observation; numerical arrays and the
original timestep controller are unchanged. The qualified original is retained.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


HELPER = r'''
// Read-only accepted-state probes. Four independent reductions, no atomics
// and no writes to any simulator field. Kernel normalization cancels in p/W.
__global__ void accepted_probe_observer(const vect* x, const double* p,
    const int* bc, const int n, const double support, double* output) {
    const double heights[4] = {0.003, 0.015, 0.030, 0.080};
    const int probe = blockIdx.x, lane = threadIdx.x;
    __shared__ double weight[256], pressure[256], count[256];
    double sw = 0.0, sp = 0.0, sc = 0.0;
    for (int i = lane; i < n; i += blockDim.x) {
        if (bc[i] != fluidParticle) continue;
        const double dx = x[i].x - 1.61, dy = x[i].y - heights[probe];
        const double q = sqrt(dx*dx + dy*dy) / support;
        if (q < 1.0) {
            const double a = 1.0-q, a2 = a*a;
            const double w = a2*a2*(1.0+4.0*q);
            sw += w; sp += w*p[i]; sc += 1.0;
        }
    }
    weight[lane]=sw; pressure[lane]=sp; count[lane]=sc;
    __syncthreads();
    for (int offset=128; offset; offset/=2) {
        if (lane < offset) {
            weight[lane]+=weight[lane+offset];
            pressure[lane]+=pressure[lane+offset];
            count[lane]+=count[lane+offset];
        }
        __syncthreads();
    }
    if (!lane) {
        output[3*probe] = weight[0] > 0.0 ? pressure[0]/weight[0] : 0.0;
        output[3*probe+1] = weight[0]; output[3*probe+2] = count[0];
    }
}

static void observe_accepted(const vect* x, const double* p, const int* bc,
    int n, double support, int step, double time) {
    const char* path = std::getenv("DYNGPU_ACCEPTED_PROBES");
    if (!path) return;
    static FILE* file = nullptr;
    static double* device = nullptr;
    if (!file) {
        file = fopen(path, "w");
        if (!file) { perror(path); exit(5); }
        checkCudaErrors(cudaMalloc((void**)&device, 12*sizeof(double)));
        fprintf(file, "step,t,p1,p2,p3,p4,w1,w2,w3,w4,n1,n2,n3,n4\n");
    }
    accepted_probe_observer<<<4,256>>>(x,p,bc,n,support,device);
    checkCudaErrors(cudaGetLastError());
    double host[12];
    checkCudaErrors(cudaMemcpy(host,device,sizeof(host),cudaMemcpyDeviceToHost));
    fprintf(file, "%d,%.17g", step,time);
    for (int field=0; field<3; ++field)
        for (int j=0;j<4;++j) {
            if (!std::isfinite(host[3*j+field])) exit(6);
            fprintf(file, ",%.17g",host[3*j+field]);
        }
    fprintf(file,"\n"); fflush(file);
}
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit('Output must be new')
    shutil.copytree(args.base, args.out)
    path = args.out/'build/src/gpusph3d.cu'
    original = path.read_text()
    marker = '__global__ void\nGetYarray'
    assert original.count(marker) == 1
    changed = original.replace(marker, HELPER+'\n'+marker, 1)
    marker = '\tconst int conv_hist_l = 5;'
    assert changed.count(marker) == 1
    changed = changed.replace(marker,
        '    observe_accepted(grdnp1_d,P_np1_d,bc_d,npts,2.0*sr,0,act_time);\n'+marker, 1)
    marker = '        // Reference instrumentation: observation and explicit bounded-run stop only.'
    assert changed.count(marker) == 1
    changed = changed.replace(marker,
        '        observe_accepted(grdnp1_d,P_np1_d,bc_d,npts,2.0*sr,tstep,act_time);\n'+marker, 1)
    path.write_text(changed)
    patch = args.out/'accepted-observer.patch'
    patch.write_text(''.join(difflib.unified_diff(original.splitlines(True),
        changed.splitlines(True), fromfile='a/gpusph3d.cu', tofile='b/gpusph3d.cu')))
    with (args.out/'observer-build.log').open('w') as log:
        subprocess.run(['make','gpuSph2d'], cwd=args.out/'build',
            stdout=log,stderr=subprocess.STDOUT,check=True)
    receipt = json.loads((args.out/'build-manifest.json').read_text())
    receipt.update(observation_base=str(args.base.resolve()),
        observation_patch_sha256=sha(patch), observation='accepted post-shift probes every physical step; no numerical changes',
        binaries={'gpuSph2d':sha(args.out/'build/gpuSph2d')})
    receipt['patched_source_sha256']={p.name:sha(p) for p in path.parent.iterdir() if p.is_file()}
    (args.out/'build-manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('Built observation-only dynGPUSPH:', args.out/'build/gpuSph2d',flush=True)


if __name__ == '__main__':
    main()
