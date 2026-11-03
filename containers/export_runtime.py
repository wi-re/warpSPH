"""Extract the NGC Torch/CUDA runtime into a clean OS stage.

Keep shared-library families (including dynamically loaded CUDA/MKL/HPC plugins),
not development archives, headers, profilers or unrelated training packages.
This recipe targets the Ubuntu 24.04 / Python 3.12 NGC 26.01 release pair.
"""
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess


def allocated_sections(path):
    """Hash ELF64 loadable section contents/addresses before stripping symbols."""
    with path.open('rb') as file:
        header = file.read(64)
        if header[:6] != b'\x7fELF\x02\x01':
            raise ValueError('Expected a little-endian ELF64 runtime library')
        offset = struct.unpack_from('<Q', header, 40)[0]
        size, count, names_index = struct.unpack_from('<HHH', header, 58)
        file.seek(offset)
        sections = [struct.unpack_from('<IIQQQQIIQQ', file.read(size)) for _ in range(count)]
        names = sections[names_index]
        file.seek(names[4])
        strings = file.read(names[5])
        result = {}
        for section in sections:
            name, kind, flags, address, start, length, _, _, alignment, _ = section
            if not flags & 2:  # SHF_ALLOC
                continue
            name = strings[name:].split(b'\0', 1)[0].decode()
            digest = hashlib.sha256()
            if kind != 8:  # SHT_NOBITS: zero-initialized memory, no file bytes
                file.seek(start)
                remaining = length
                while remaining:
                    chunk = file.read(min(remaining, 8 * 1024 * 1024))
                    if not chunk:
                        raise ValueError('Truncated ELF section')
                    digest.update(chunk)
                    remaining -= len(chunk)
            result[name] = (kind, flags, address, length, alignment, digest.hexdigest())
        return result


def export_alias(path, destination):
    """Rebase an alias onto exported files, without distro alternatives."""
    target = destination / str(path).lstrip('/')
    resolved = destination / str(path.resolve()).lstrip('/')
    if resolved.exists() and not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(os.path.relpath(resolved, target.parent.resolve()))
        if not target.resolve().is_relative_to(destination.resolve()):
            raise RuntimeError(f'Alias escaped the runtime staging root: {target}')


def export(destination):
    destination = Path(destination)
    copied = set()
    os_packages = set()
    optional_unresolved = {}
    stripped = {}
    seen_dependencies = set()

    def copy(path, target=None):
        path = Path(path)
        if not path.is_file():
            return
        target = destination / str(target or path).lstrip('/')
        target.parent.mkdir(parents=True, exist_ok=True)
        # Dereference: no dangling links to development-stage-only paths.
        if target.exists():
            return
        shutil.copy2(path, target)
        copied.add((path, target))

    if platform.python_version_tuple()[:2] != ('3', '12'):
        raise RuntimeError('Runtime recipe requires the Python 3.12 base pair')
    release = Path('/etc/os-release').read_text()
    if 'ID=ubuntu\n' not in release or 'VERSION_ID="24.04"' not in release:
        raise RuntimeError('Runtime recipe requires the Ubuntu 24.04 base pair')
    site = Path('/opt/venv/lib/python3.12/site-packages')
    torch = metadata.distribution('torch')
    module_roots = []
    # functorch is part of Torch's distribution; Triton is needed by the
    # simulator's optional torch.compile wall operators and core imports.
    for distribution in (torch, metadata.distribution('pytorch-triton')):
        names = {path.parts[0] for path in distribution.files if path.parts[0] != '..'}
        for name in sorted(names):
            source = Path(distribution.locate_file(name))
            if source.is_dir():
                module_roots.append(source)
                for path in source.rglob('*'):
                    if '__pycache__' not in path.parts and path.suffix != '.pyc':
                        copy(path, site / path.relative_to(source.parent))
    # Triton's launcher JIT needs CUDA headers, but not nvcc or static archives.
    for path in Path('/usr/local/cuda/include').rglob('*'):
        if path.is_file():
            copy(path, path.resolve())
    # The accepted Isambard deployment maps libcuda from this forward-compat
    # bundle. Preserve the vendor's activation check, never driver stubs.
    for path in Path('/usr/local/cuda/compat').rglob('*'):
        if path.is_file():
            copy(path, path.resolve())
    copy('/etc/shinit_v2')
    copy('/usr/local/bin/cudaCheck')
    copy('/usr/local/cuda/bin/ptxas', Path('/usr/local/cuda/bin/ptxas').resolve())
    print('Torch/Triton modules and CUDA headers exported', flush=True)

    # CUDA/cuDNN and MKL use dlopen as well as ELF dependencies. Retain their
    # complete shared-library families, including hardware-specific dispatch.
    patterns = [
        '/usr/local/cuda/targets/*/lib/**/*.so*',
        '/usr/local/cuda/lib64/**/*.so*',
        '/usr/lib/*/libcudnn*.so*', '/usr/lib/*/libnccl*.so*',
        '/usr/local/lib/libmkl*.so*', '/usr/local/lib/libiomp*.so*',
        '/opt/hpcx/**/lib/**/*.so*',
    ]
    libraries = set()
    for pattern in patterns:
        for path in Path('/').glob(pattern.lstrip('/')):
            if path.is_file() and 'stubs' not in path.parts and not path.name.startswith('libmkl_sycl'):
                # Do not flatten symlink aliases into duplicate huge binaries.
                libraries.add(path.resolve())
    for path in sorted(libraries):
        copy(path)
    # Resolve transitive dependencies from the base, retaining vendor libraries
    # outside these trees. OS ABI libraries come from apt in the final stage.
    roots = libraries | {path for root in module_roots for path in root.rglob('*.so*')} | {
        Path('/usr/local/bin/cudaCheck'), Path('/usr/local/cuda/bin/ptxas')}
    for path in sorted(roots):
        if not path.is_file():
            continue
        output = subprocess.check_output(['ldd', str(path)], text=True, stderr=subprocess.DEVNULL)
        for line in output.splitlines():
            fields = line.split()
            if 'not found' in line:
                # The NVIDIA container runtime / Apptainer supplies the host
                # driver, or the explicitly retained forward-compat bundle.
                if fields[0] == 'libcuda.so.1' or fields[0].startswith('libnvidia-'):
                    continue
                if str(path).startswith('/opt/hpcx/'):
                    # Some optional fabric plugins in NGC already require
                    # site-provided libraries (e.g. XPMEM). Record, don't mask.
                    optional_unresolved.setdefault(str(path), []).append(fields[0])
                    continue
                raise RuntimeError(f'Unresolved base library in {path}: {line}')
            if '=>' not in fields or not fields[2].startswith('/'):
                continue
            dependency = Path(fields[2]).resolve()
            if dependency in seen_dependencies:
                continue
            seen_dependencies.add(dependency)
            if str(dependency).startswith(('/usr/local/', '/opt/')) or dependency.name.startswith(
                    ('libnccl', 'libcudnn', 'libnv', 'libibverbs', 'librdmacm', 'libgdrapi')):
                if not any(root in dependency.parents for root in module_roots):
                    copy(dependency)
            else:
                candidates = [str(dependency), fields[2]]
                if str(dependency).startswith('/usr/lib/'):
                    candidates.append(str(dependency)[4:])
                owner = None
                for candidate in candidates:
                    result = subprocess.run(['dpkg-query', '-S', candidate], capture_output=True, text=True)
                    if result.returncode == 0:
                        owner = result.stdout.split(': ', 1)[0].split(':')[0]
                        break
                if not owner or not re.fullmatch(r'[a-z0-9][a-z0-9+.-]*', owner):
                    raise RuntimeError(f'No OS package owner for {dependency}')
                os_packages.add(owner)

    # Re-create aliases pointing only to files actually exported, preserving
    # SONAMEs, CUDA directory aliases and HPC plugin-relative lookups.
    aliases = [Path('/usr/local/cuda'), Path('/usr/local/cuda-13')]
    aliases += list(Path('/usr/local/cuda').rglob('*'))
    aliases += list(Path('/opt/hpcx').rglob('*'))
    aliases += list(Path('/usr/lib').glob('*/libcudnn*.so*'))
    aliases += list(Path('/usr/lib').glob('*/libnccl*.so*'))
    aliases += list(Path('/usr/local/lib').glob('libmkl*.so*'))
    for path in aliases:
        if path.is_symlink():
            export_alias(path, destination)
    print('Shared-library closure and aliases exported', flush=True)

    # Preserve available vendor notices and the package's own license files.
    for directory in ('/licenses', '/usr/share/licenses', '/opt/nvidia'):
        root = Path(directory)
        if root.exists():
            for path in root.rglob('*'):
                if path.is_file() and any(word in path.name.lower() for word in ('license', 'copyright', 'notice')):
                    copy(path)
    for path in Path('/').glob('*LICENSE*'):
        copy(path)
    for path in Path('/usr/local/cuda').rglob('*'):
        if path.is_file() and any(word in path.name.lower() for word in ('license', 'eula', 'copyright', 'notice')):
            copy(path, path.resolve())
    for path in Path('/usr/share/doc').glob('*'):
        if any(name in path.name for name in ('cuda', 'cublas', 'cudnn', 'nccl', 'nvshmem', 'cusparse', 'cusolver', 'cufft', 'curand')):
            for file in path.rglob('*'):
                copy(file)
    # NGC's Triton and some numerical libraries retain large native symbol
    # tables. Strip only when every allocated section remains byte-identical.
    # Keep original hashes as well as the transformed runtime-file hashes.
    for source, target in sorted(copied):
        if '.so' not in target.name or 'compat' in target.parts or target.stat().st_size < 1024 * 1024:
            continue
        with target.open('rb') as file:
            if file.read(4) != b'\x7fELF':
                continue
        temporary = target.with_name(target.name + '.stripped')
        shutil.copy2(target, temporary)
        subprocess.run(['strip', '--strip-unneeded', str(temporary)], check=True)
        if allocated_sections(target) != allocated_sections(temporary):
            temporary.unlink()
            continue
        before = target.stat().st_size
        after = temporary.stat().st_size
        if after < before:
            with target.open('rb') as file:
                original_hash = hashlib.file_digest(file, 'sha256').hexdigest()
            stripped[str(target.relative_to(destination))] = dict(
                sourceSha256=original_hash, originalBytes=before, runtimeBytes=after,
                allocatedSectionsUnchanged=True)
            temporary.replace(target)
        else:
            temporary.unlink()
    Path('/runtime-os-packages.txt').write_text('\n'.join(sorted(os_packages)) + '\n')
    manifest = dict(architecture=platform.machine(), python=platform.python_version(),
                    osPackages=sorted(os_packages), optionalUnresolved=optional_unresolved,
                    strippedLibraries=stripped,
                    aliases={str(path.relative_to(destination)): str(path.readlink())
                             for path in destination.rglob('*') if path.is_symlink()},
                    torch=metadata.version('torch'), triton=metadata.version('pytorch-triton'), files={
                        str(target.relative_to(destination)): hashlib.file_digest(target.open('rb'), 'sha256').hexdigest()
                        for _, target in sorted(copied)})
    target = destination / 'opt/warpsph/runtime-export.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print('Runtime inventory:', len(copied), 'files;',
          sum(item['originalBytes'] - item['runtimeBytes'] for item in stripped.values()),
          'symbol bytes removed with allocated sections unchanged', flush=True)


if __name__ == '__main__':
    import sys
    export(sys.argv[1])
