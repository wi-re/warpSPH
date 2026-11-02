#!/usr/bin/env bash
# Container-side native build. Invoked only by build_network.slurm.
set -euo pipefail
export HOME=/opt/slingshot TMPDIR=/tmp
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH="/host/libfabric/lib64:${LD_LIBRARY_PATH:-}:/host/usr/lib64"
export PKG_CONFIG_PATH="/host/libfabric/lib64/pkgconfig:${PKG_CONFIG_PATH:-}"
[[ $(uname -m) == aarch64 ]] || { echo 'Build on native Isambard ARM64.' >&2; exit 2; }
for tool in gcc g++ make tar python; do command -v "$tool" >/dev/null; done
[[ -f /host/libfabric/include/rdma/fabric.h && -f $CUDA_HOME/include/cuda.h ]] || exit 2
# Check the published release checksums before any extraction/configuration.
python - <<'SOURCES_PY'
import hashlib, json
from pathlib import Path
lock = json.loads(Path('/opt/slingshot/source-lock.json').read_text())
for package in lock.values():
    path = Path('/sources') / package['file']
    algorithm = 'sha512' if 'sha512' in package else 'sha256'
    assert hashlib.new(algorithm, path.read_bytes()).hexdigest() == package[algorithm], 'Source checksum mismatch: '+str(path)
    print('Verified source:', package['file'], flush=True)
SOURCES_PY
mkdir /opt/slingshot/src
cd /opt/slingshot/src
tar -xzf /sources/hwloc-2.13.0.tar.gz
tar -xzf /sources/aws-ofi-nccl-1.18.0.tar.gz
cd hwloc-2.13.0
./configure --prefix=/opt/slingshot/hwloc --libdir=/opt/slingshot/hwloc/lib \
  --disable-nvml --disable-cuda --disable-opencl --disable-rsmi --disable-levelzero \
  --disable-plugins --disable-libxml2 --disable-libudev --disable-cairo
make -j "${BUILD_JOBS:-8}"
make install
cd ../aws-ofi-nccl-1.18.0
export LD_LIBRARY_PATH="/opt/slingshot/hwloc/lib:$LD_LIBRARY_PATH"
./configure --prefix=/opt/slingshot/aws-ofi-nccl --libdir=/opt/slingshot/aws-ofi-nccl/lib \
  --with-cuda="$CUDA_HOME" --with-libfabric=/host/libfabric \
  --with-hwloc=/opt/slingshot/hwloc --disable-platform-aws --disable-tests
make -j "${BUILD_JOBS:-8}"
make install
source /opt/slingshot/network-setup.sh
ldd /opt/slingshot/aws-ofi-nccl/lib/libnccl-net.so | tee /opt/slingshot/plugin-ldd.txt
if grep -q 'not found' /opt/slingshot/plugin-ldd.txt; then exit 1; fi
/host/libfabric/bin/fi_info -p cxi > /opt/slingshot/cxi-provider.txt
python - <<'NATIVE_PY'
import ctypes, hashlib, json, os, platform, subprocess
from pathlib import Path
import torch
assert platform.machine() == 'aarch64'
ctypes.CDLL('/opt/slingshot/aws-ofi-nccl/lib/libnccl-net.so', mode=os.RTLD_LOCAL | os.RTLD_NOW)
paths = {line.split(maxsplit=5)[5] for line in Path('/proc/self/maps').read_text().splitlines()
         if len(line.split(maxsplit=5)) == 6 and line.split(maxsplit=5)[5].startswith('/')}
nccl = sorted(p for p in paths if Path(p).name.startswith('libnccl.so'))
assert len(nccl) == 1, nccl
lib = ctypes.CDLL(nccl[0], mode=os.RTLD_LOCAL | os.RTLD_NOLOAD)
lib.ncclGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
lib.ncclGetVersion.restype = ctypes.c_int
version = ctypes.c_int()
assert lib.ncclGetVersion(ctypes.byref(version)) == 0
accepted = json.loads(Path('/build/accepted-image.json').read_text())
assert version.value == accepted['ncclRuntime'], 'Build replaced the accepted NCCL runtime'
assert not nccl[0].startswith('/host/'), nccl
files = {str(p.relative_to('/opt/slingshot')): hashlib.sha256(p.read_bytes()).hexdigest()
         for folder in ['aws-ofi-nccl/lib', 'hwloc/lib']
         for p in sorted(Path('/opt/slingshot', folder).glob('*.so*')) if p.is_file() and not p.is_symlink()}
record = dict(status='passed', architecture=platform.machine(), jobId=os.environ['QA_BUILD_JOB_ID'],
              acceptedImage=accepted, sources=json.loads(Path('/opt/slingshot/source-lock.json').read_text()),
              torch=torch.__version__, cuda=torch.version.cuda, ncclRuntime=version.value,
              ncclLibraries=nccl, compiler=subprocess.check_output(['gcc','--version'],text=True).splitlines()[0],
              mappedCommunicationLibraries=sorted(p for p in paths if Path(p).name.startswith(('libnccl','libcuda','libfabric','libcxi','libhwloc'))),
              librarySha256=files)
Path('/opt/slingshot/native-build.json').write_text(json.dumps(record, indent=2, sort_keys=True)+'\n')
print(json.dumps(record, indent=2), flush=True)
NATIVE_PY
echo 'OFI build/load PASSED. Repeat the GPU network preflight before two nodes.'
