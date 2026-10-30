#!/usr/bin/env python3
"""Add read-only RK/operator observations to an already qualified isolated build.

Copies build objects, recompiles HashFunctions, and records an instrumentation
diff and binary receipt. DYNGPU_AUDIT_DIR enables first-step iterations 0,1,199.
The original checkout and qualified binary are never changed.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit('Output must be new')
    shutil.copytree(args.base, args.out)
    src = args.out / 'build/src'
    p = src / 'HashFunctions.cu'
    original = p.read_text()
    helper = r'''
#include <cstdio>
#include <cstdlib>
static void ac_audit_dump(const char* root, int iteration, int stage,
                          const char* name, const void* ptr, size_t bytes) {
    std::vector<unsigned char> host(bytes);
    checkCudaErrors(cudaMemcpy(host.data(), ptr, bytes, cudaMemcpyDeviceToHost));
    char path[4096];
    snprintf(path, sizeof(path), "%s/i%03d-s%d-%s.bin", root, iteration, stage, name);
    FILE* file = fopen(path, "wb");
    if (!file || fwrite(host.data(), 1, bytes, file) != bytes) abort();
    fclose(file);
}
'''
    s = original.replace('extern "C" //Why?', helper + '\nextern "C" //Why?', 1)
    begin = s.index('\tvoid ResidualHash3(')
    end = s.index('\n\tvoid ShiftHash(', begin)
    part = s[begin:end]
    hook = '''
        static int audit_step = -1;
        if (isubit == 0 && RKsubit == 0) ++audit_step;
        const char* audit_root = getenv("DYNGPU_AUDIT_DIR");
        bool audit = audit_root && audit_step == 0 && (isubit == 0 || isubit == 1 || isubit == 199);
        double* audit_values = nullptr;
        if (audit) {
            checkCudaErrors(cudaMalloc((void**)&audit_values, npts * 10 * sizeof(double)));
            checkCudaErrors(cudaMemset(audit_values, 0, npts * 10 * sizeof(double)));
            checkCudaErrors(cudaMemcpyToSymbol(ac_audit_values, &audit_values, sizeof(audit_values)));
        }
'''
    part = part.replace('\n\t\t// thread per particle', hook + '\n\t\t// thread per particle', 1)
    fields = {'index': ('gridParticleIndex', 'uint'), 'x': ('SortedPos', 'vect'),
              'v': ('SortedVel', 'vect'), 'p': ('SortedP', 'double'),
              'rho': ('SortedRho', 'double'), 'm': ('Sortedm', 'double'),
              'bc': ('Sortedbc', 'int'), 'lambda': ('isurf', 'double'),
              'gradP': ('renormPgrad', 'vect'), 'norm': ('norm', 'vect'),
              'divVfluid': ('divV_sorted_d', 'double'), 'lapV': ('velLap_d', 'vect'),
              'wallV': ('velBC', 'vect'), 'rV': ('RVnp1', 'vect'),
              'rP': ('Rrhonp1', 'double')}
    dump = '\n        if (audit) {\n'
    for name, (pointer, dtype) in fields.items():
        dump += f'            ac_audit_dump(audit_root, isubit, RKsubit, "{name}", {pointer}, npts * sizeof({dtype}));\n'
    dump += '''            ac_audit_dump(audit_root, isubit, RKsubit, "terms", audit_values, npts * 10 * sizeof(double));
            checkCudaErrors(cudaFree(audit_values));
            audit_values = nullptr;
            checkCudaErrors(cudaMemcpyToSymbol(ac_audit_values, &audit_values, sizeof(audit_values)));
            printf("REFERENCE AUDIT iteration=%d stage=%d dtau=%.17g beta=%.17g\\n", isubit, RKsubit, dTau, RK_ps_CFL * sr / dTau);
            fflush(stdout);
        }
'''
    pos = part.rfind('\n\t}')
    part = part[:pos] + dump + part[pos:]
    s = s[:begin] + part + s[end:]
    begin = s.index('\tvoid ShiftHash(')
    end = s.index('\n\tdouble getTR(', begin)
    part = s[begin:end]
    hook = '''
        static int shift_audit_step = 0;
        const char* shift_audit_root = getenv("DYNGPU_AUDIT_DIR");
        if (shift_audit_root && shift_audit_step < 3) {
            checkCudaErrors(cudaDeviceSynchronize());
'''
    for name, pointer, dtype in (
        ('index', 'gridParticleIndex', 'uint'), ('x_pre', 'SortedPos', 'vect'),
        ('v_pre', 'SortedVel', 'vect'), ('p_pre', 'SortedP', 'double'),
        ('x_post', 'newPos', 'vect'), ('v_post', 'newVel', 'vect'),
        ('p_post', 'newP', 'double'), ('deltaU_BDF', 'deltaU_BDF', 'vect')):
        hook += f'            ac_audit_dump(shift_audit_root, shift_audit_step, 9, "{name}", {pointer}, npts * sizeof({dtype}));\n'
    hook += '        }\n        ++shift_audit_step;\n'
    part = part.replace('\n\t\treturn;', hook + '\n\t\treturn;', 1)
    s = s[:begin] + part + s[end:]
    p.write_text(s)
    h = src / 'HashFunctions_cuda.cuh'
    horiginal = h.read_text()
    s = horiginal.replace('    __global__\n        void residualHash2D',
        '    __device__ double* ac_audit_values = nullptr;\n\n    __global__\n        void residualHash2D', 1)
    assert s != horiginal
    s = s.replace('            // JST Viscosities', '            vect_t auditBeforeJst = Resv;\n            // JST Viscosities', 1)
    s = s.replace('            // Surface wall damping', '            vect_t auditJst = Resv - auditBeforeJst;\n            // Surface wall damping', 1)
    needle = '            Rrhonp1[originalIndex] = ResP;'
    pos = s.index(needle, s.index('void residualHash2D')) + len(needle)
    hook = '''
            if (ac_audit_values) {
                double* a = ac_audit_values + originalIndex * 10;
                a[0] = irhoA * PGti.x; a[1] = irhoA * PGti.y;
                a[2] = VISC.x; a[3] = VISC.y;
                a[4] = auditJst.x; a[5] = auditJst.y;
                a[6] = beta * sr * divVsum.x; a[7] = beta * sr * divVsum.y;
                a[8] = roc; a[9] = Laplacian;
            }
'''
    s = s[:pos] + hook + s[pos:]
    h.write_text(s)
    diff = ''.join(difflib.unified_diff(original.splitlines(True), p.read_text().splitlines(True), fromfile='a/HashFunctions.cu', tofile='b/HashFunctions.cu'))
    diff += ''.join(difflib.unified_diff(horiginal.splitlines(True), h.read_text().splitlines(True), fromfile='a/HashFunctions_cuda.cuh', tofile='b/HashFunctions_cuda.cuh'))
    (args.out / 'audit.patch').write_text(diff)
    # Header dependency would rebuild every unit; only HashFunctions consumes
    # the modified internal CUDA header. All remaining objects are qualified.
    with (args.out / 'audit-build.log').open('w') as log:
        subprocess.run(['make', '-o', 'src/HashFunctions_cuda.cuh', 'gpuSph2d'], cwd=args.out/'build', stdout=log, stderr=subprocess.STDOUT, check=True)
    receipt = json.loads((args.out/'build-manifest.json').read_text())
    receipt['observation_base'] = str(args.base.resolve())
    receipt['observation_patch_sha256'] = sha(args.out/'audit.patch')
    receipt['binaries'] = {'gpuSph2d': sha(args.out/'build/gpuSph2d')}
    receipt['patched_source_sha256'] = {p.name: sha(p) for p in src.iterdir() if p.is_file()}
    (args.out/'build-manifest.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print('Built observational binary', args.out/'build/gpuSph2d', flush=True)


if __name__ == '__main__':
    main()
