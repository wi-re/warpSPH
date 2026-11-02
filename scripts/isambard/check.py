#!/usr/bin/env python3
"""CPU-only checkpoint comparison and retrieved-artifact verification (NumPy/h5py)."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_digest(file):
    """Native rowChunkCheckpointVersion1 checksum; no Torch import needed.

    This is the serialization contract in warpSPH/io/chunkCheckpoint.py:_fileDigest.
    Compare physical fields separately: valid files may differ in ownership/cache.
    """
    digest = hashlib.sha256()
    def plain(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value.decode() if isinstance(value, bytes) else value
    def update(value):
        digest.update(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
    def visit(name, obj):
        update((name, {key: plain(value) for key, value in obj.attrs.items() if key != 'contentChecksum'}))
        if isinstance(obj, h5py.Dataset):
            value = np.asarray(obj[()])
            update((value.dtype.str, list(value.shape)))
            digest.update(value.tobytes(order='C'))
    visit('', file)
    file.visititems(visit)
    return digest.hexdigest()


def validate_checkpoint(file):
    if file.attrs.get('rowChunkCheckpointVersion') != 1 or file.attrs.get('complete') != 1:
        raise ValueError('Unsupported or incomplete native checkpoint')
    if file.attrs.get('contentChecksum') != checkpoint_digest(file):
        raise ValueError('Native checkpoint checksum mismatch')
    for group in ('state', 'clock', 'forcing', 'context'):
        if group not in file:
            raise ValueError('Missing checkpoint group: ' + group)
    identifiers = file['state/UIDs'][()]
    if identifiers.ndim != 1 or len(np.unique(identifiers)) != len(identifiers):
        raise ValueError('Particle UIDs must be unique')
    def finite(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.dtype.kind in 'fc' and not np.isfinite(obj[()]).all():
            raise ValueError('Nonfinite checkpoint field: ' + name)
    file.visititems(finite)


def compare(reference, candidate, *, rtol32=2e-5, atol32=2e-8, rtol64=1e-10, atol64=1e-12):
    for value in (rtol32, atol32, rtol64, atol64):
        if not np.isfinite(value) or value < 0:
            raise ValueError('Comparison tolerances must be finite and nonnegative')
    result = {'reference': str(reference), 'candidate': str(candidate), 'fields': {}, 'passed': True,
              'tolerances': dict(rtol32=rtol32, atol32=atol32, rtol64=rtol64, atol64=atol64)}
    with h5py.File(reference) as first, h5py.File(candidate) as second:
        validate_checkpoint(first)
        validate_checkpoint(second)
        for key in ('physicalSignature', 'completedSteps', 'time', 'stateClass'):
            if first.attrs[key] != second.attrs[key]:
                raise ValueError('Checkpoint attribute differs: ' + key)
        order_a = np.argsort(first['state/UIDs'][()])
        order_b = np.argsort(second['state/UIDs'][()])
        for group in ('state', 'forcing', 'clock', 'context'):
            if set(first[group]) != set(second[group]):
                raise ValueError('Checkpoint field set differs: ' + group)
            for name in first[group]:
                a, b = first[group][name][()], second[group][name][()]
                if a.shape != b.shape or a.dtype != b.dtype:
                    raise ValueError('Checkpoint shape/dtype differs: ' + group + '/' + name)
                if group == 'state' and name != 'sourceEnergyBudget' and a.ndim and a.shape[0] == len(order_a):
                    a, b = a[order_a], b[order_b]
                exact = group in ('forcing', 'clock') or a.dtype.kind not in 'fc'
                rtol, atol = (0., 0.) if exact else ((rtol32, atol32) if a.dtype.itemsize <= 4 else (rtol64, atol64))
                matches = np.array_equal(a, b) if exact else np.allclose(a, b, rtol=rtol, atol=atol)
                delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
                field = dict(passed=bool(matches), exact=exact, maxAbsolute=float(delta.max()) if delta.size else 0., rtol=rtol, atol=atol)
                result['fields'][group + '/' + name] = field
                result['passed'] &= bool(matches)
    return result


def seal(directory):
    directory = Path(directory)
    inventory = {}
    # Slurm's live stdout and transport logs are deliberately outside this set.
    for name in ('configs', 'runs', 'probes', 'checks'):
        for path in sorted((directory / name).rglob('*')):
            if path.is_symlink():
                raise ValueError('Artifact symlinks are not supported')
            if path.is_file():
                inventory[path.relative_to(directory).as_posix()] = file_hash(path)
    inventory['report.json'] = file_hash(directory / 'report.json')
    atomic_json(directory / 'artifacts.json', {'formatVersion': 1, 'sha256': inventory})
    return inventory


def verify(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / 'artifacts.json').read_text())
    if manifest.get('formatVersion') != 1 or not manifest.get('sha256'):
        raise ValueError('Missing or unsupported artifact inventory')
    for name, expected in manifest['sha256'].items():
        path = directory / name
        if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink() or directory not in path.resolve().parents:
            raise ValueError('Unsafe artifact path: ' + name)
        if file_hash(path) != expected:
            raise ValueError('Missing or damaged retrieved file: ' + name)
        if path.suffix == '.h5':
            with h5py.File(path) as file:
                validate_checkpoint(file)
    report = json.loads((directory / 'report.json').read_text())
    if report.get('status') != 'passed':
        raise ValueError('Retrieved campaign did not pass: ' + report.get('status', 'unknown'))
    return {'status': 'passed', 'stage': report['stage'], 'filesVerified': len(manifest['sha256'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    comparison = commands.add_parser('compare')
    comparison.add_argument('reference', type=Path)
    comparison.add_argument('candidate', type=Path)
    comparison.add_argument('--output', type=Path, required=True)
    for name, default in (('rtol32', 2e-5), ('atol32', 2e-8), ('rtol64', 1e-10), ('atol64', 1e-12)):
        comparison.add_argument('--' + name, type=float, default=default)
    for name in ('seal', 'verify'):
        commands.add_parser(name).add_argument('directory', type=Path)
    args = parser.parse_args()
    if args.command == 'compare':
        result = compare(args.reference, args.candidate, **{key: getattr(args, key) for key in ('rtol32', 'atol32', 'rtol64', 'atol64')})
        atomic_json(args.output, result)
        if not result['passed']:
            raise SystemExit('Checkpoint comparison failed; inspect ' + str(args.output))
        print('Checkpoint comparison passed:', args.output, flush=True)
    elif args.command == 'seal':
        print('Sealed', len(seal(args.directory)), 'stable artifacts', flush=True)
    else:
        print(json.dumps(verify(args.directory)), flush=True)


if __name__ == '__main__':
    main()
