#!/usr/bin/env python3
"""Build a pinned dynGPUSPH ACSPH reference without modifying its checkout.

CUDA 13 / Blackwell recipe. Requires CUDA toolkit, GCC 13 and Eigen. Download
headers with --download on first use. Each output root is immutable once built.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request

HERE=Path(__file__).resolve().parent
SOURCES='gpusph3d.cu HashFunctions.cu boundary.cu particles.cu cfl.cu dynbound.cu plotting.cpp parameters.cpp common_functions.cpp fluid.cpp rbf.cpp'.split()

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=Path.home()/'dev/dynGPUSPH')
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--cuda',type=Path,default=Path.home()/'miniconda3/envs/warp')
    ap.add_argument('--eigen',type=Path,default=Path.home()/'dev/SPlisHSPlasH/extern/eigen')
    ap.add_argument('--cxx',default='/usr/bin/g++-13')
    ap.add_argument('--arch',default='sm_120')
    ap.add_argument('--download',action='store_true')
    ap.add_argument('--jobs',type=int,default=4)
    args=ap.parse_args()
    root=args.out.resolve(); deps=root/'deps'; build=root/'build'; src=build/'src'
    if src.exists(): raise SystemExit(f'Refusing to replace existing source: {src}')
    commit=subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip()
    if not commit.startswith('2dbc27a'): raise SystemExit(f'Expected paper-era 2dbc27a, found {commit}')
    expected=json.loads((HERE/'dependencies.json').read_text())
    deps.mkdir(parents=True,exist_ok=True)
    for name,digest in expected.items():
        path=deps/name
        if not path.exists():
            if not args.download: raise SystemExit(f'Missing {path}; use --download')
            url=(f'https://raw.githubusercontent.com/jlblancoc/nanoflann/v1.4.3/include/{name}' if name=='nanoflann.hpp' else f'https://raw.githubusercontent.com/NVIDIA/cuda-samples/v12.9/Common/{name}')
            print('Download',url,flush=True); path.write_bytes(urllib.request.urlopen(url,timeout=30).read())
        if sha(path)!=digest: raise SystemExit(f'Header hash mismatch: {path}')
    originals={p.name:sha(p) for p in (args.source/'src').iterdir() if p.is_file()}
    # Eigen is external to dynGPUSPH: retain a complete content fingerprint.
    eigen_hash=hashlib.sha256()
    for p in sorted((args.eigen/'Eigen').rglob('*')):
        if p.is_file(): eigen_hash.update(str(p.relative_to(args.eigen)).encode()); eigen_hash.update(p.read_bytes())
    for path in deps.glob('helper*.h'):
        path.write_text(path.read_text().replace('<exception.h>','<exception>'))
    shutil.copytree(args.source/'src',src)
    subprocess.run(['patch','--batch','--fuzz=0','-p1','-i',str(HERE/'cuda13.patch')],cwd=src,check=True)
    shutil.copyfile(HERE/'plotting.cpp',src/'plotting.cpp')
    shutil.copyfile(HERE/'range.hpp',deps/'range.hpp')
    (deps/'TECIO.h').write_text('#pragma once\nusing INTEGER4 = int;\n')
    cuda=args.cuda.resolve()
    flags=f'-std=c++17 -O3 -use_fast_math -arch={args.arch} -ccbin {args.cxx} -DDOF2=1 -DLINEARPG -DEIGEN_DONT_PARALLELIZE -DEIGEN_DONT_VECTORIZE -DEIGEN_DEFAULT_DENSE_INDEX_TYPE=int'
    inc=f'-I../deps -Isrc -I{args.eigen.resolve()} -I{cuda}/targets/x86_64-linux/include/cccl'
    link=f'-L{cuda}/targets/x86_64-linux/lib -Xlinker -rpath -Xlinker {cuda}/targets/x86_64-linux/lib -lcublas -lcusparse -Xcompiler -fopenmp'
    objects=' '.join(name+'.o' for name in SOURCES)
    mk=f'''NVCC := {cuda}/bin/nvcc
FLAGS := {flags}
INC := {inc}
LINK := {link}
OBJECTS := {objects}
HEADERS := $(wildcard src/*.h src/*.cuh ../deps/*.h ../deps/*.hpp)
all: gpuSph2d gpuSph2d-cavity
%.o: src/% $(HEADERS)
\t$(NVCC) $(FLAGS) $(INC) -x cu -c $< -o $@ > $@.log 2>&1
HashFunctions-cavity.o: src/HashFunctions.cu $(HEADERS)
\t$(NVCC) $(FLAGS) -DNOSLIP -DLDCAVITY $(INC) -x cu -c $< -o $@ > $@.log 2>&1
gpuSph2d: $(OBJECTS)
\t$(NVCC) $(FLAGS) $(OBJECTS) $(LINK) -o $@ > link.log 2>&1
gpuSph2d-cavity: $(filter-out HashFunctions.cu.o,$(OBJECTS)) HashFunctions-cavity.o
\t$(NVCC) $(FLAGS) $^ $(LINK) -o $@ > link-cavity.log 2>&1
'''
    (build/'Makefile').write_text(mk)
    manifest={'source':str(args.source.resolve()),'commit':commit,'source_sha256':originals,
              'patched_source_sha256':{p.name:sha(p) for p in src.iterdir() if p.is_file()},
              'patch_sha256':sha(HERE/'cuda13.patch'),'dependency_upstream_sha256':expected,
              'dependency_patched_sha256':{p.name:sha(p) for p in deps.iterdir() if p.is_file()},
              'eigen':str(args.eigen.resolve()),'eigen_content_sha256':eigen_hash.hexdigest(),
              'flags':flags,'cavity_additional_flags':'-DNOSLIP -DLDCAVITY (HashFunctions.cu only; only active consumers)',
              'compiler':subprocess.check_output([str(cuda/'bin/nvcc'),'--version'],text=True)}
    (root/'build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with (root/'build.log').open('w') as log:
        subprocess.run(['make','-j',str(args.jobs)],cwd=build,stdout=log,stderr=subprocess.STDOUT,check=True)
    manifest['binaries']={name:sha(build/name) for name in ('gpuSph2d','gpuSph2d-cavity')}
    (root/'build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Built',build/'gpuSph2d','and cavity variant',flush=True)

if __name__=='__main__': main()
