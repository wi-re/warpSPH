"""Freeze the completed fluid matrix and verify every archived file by SHA-256.

Creates a new versioned local archive; never overwrites one. Large raw evidence
stays outside Git. The archive can be verified without CUDA or a daemon.
"""
import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import time

from run_fluidMatrix import DEFAULT_ROOT, REPO, write_json


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(archive, inventory):
    expected = {r['path']: r for r in inventory['files']}
    checked, total, seen = 0, 0, set()
    process = subprocess.Popen(['zstd', '-q', '-d', '-c', str(archive)], stdout=subprocess.PIPE)
    with tarfile.open(fileobj=process.stdout, mode='r|') as stream:
        for member in stream:
            if not member.isfile():
                raise ValueError(f'Unexpected non-file archive member: {member.name}')
            key = str(Path(member.name).relative_to(inventory['root_name']))
            if key in seen:
                raise ValueError(f'Duplicate member: {key}')
            seen.add(key)
            if key == 'preservation/inventory.json':
                embedded = json.load(stream.extractfile(member))
                if embedded != inventory:
                    raise ValueError('Embedded inventory differs from the expected inventory')
                continue
            row = expected.get(key)
            if row is None or member.size != row['size']:
                raise ValueError(f'Unexpected member or size mismatch: {key}')
            h = hashlib.sha256()
            with stream.extractfile(member) as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                    h.update(block)
            if h.hexdigest() != row['sha256']:
                raise ValueError(f'Archived file hash mismatch: {key}')
            checked += 1
            total += member.size
            if checked % 1000 == 0:
                print(f'VERIFIED {checked}/{len(expected)} files, {total / 1e9:.2f} GB', flush=True)
    process.stdout.close()
    if process.wait() != 0:
        raise RuntimeError('zstd archive integrity check failed')
    if seen != set(expected) | {'preservation/inventory.json'}:
        raise ValueError('Archive inventory is incomplete')
    return dict(verified_files=checked, verified_bytes=total)


def preserve(root, archive):
    if archive.exists() or archive.with_suffix(archive.suffix + '.partial').exists():
        raise FileExistsError('Archive already exists; use another version or --verify')
    queue = json.loads((root / 'queue_status.json').read_text())
    audit = json.loads((root / 'final_audit.json').read_text())
    if queue['state'] != 'finished' or audit['evidence_checks_passed'] != audit['attempts']:
        raise ValueError('Only a completed, fully audited matrix may be frozen')
    keep = root / 'preservation'
    keep.mkdir(exist_ok=True)
    chosen = [REPO / 'README.md', REPO / 'pyproject.toml', REPO / 'CLAUDE.md',
              *sorted((REPO / 'docs/fluid_matrix').rglob('*')),
              REPO / 'tests/test_fluidMatrixHarness.py']
    chosen += [REPO / 'scripts' / name for name in (
        'run_fluidMatrix.py', 'analyze_fluidMatrix.py', 'audit_fluidMatrix.py',
        'plot_fluidMatrixRanking.py', 'preserve_fluidMatrix.py',
        'probe_deltaSPHMarrone.py', 'plot_sloshingOverlay.py',
        'probe_englishWedge.py', 'probe_omniEnglishWedge.py', 'probe_taylorCouette.py')]
    chosen.append(REPO / 'examples/sloshingTank/SPHERIC_TestCase10/data_files/lateral_water_1x.txt')
    for path in chosen:
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        target = keep / 'repository' / path.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    write_json(keep / 'runtime.json', dict(python=sys.version, executable=sys.executable,
        platform=platform.platform(), packages=sorted(
            [{'name': d.metadata['Name'], 'version': d.version} for d in metadata.distributions()],
            key=lambda d: (d['name'] or '').lower())))
    write_json(keep / 'git_provenance.json', dict(
        source_head_at_preservation=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPO, text=True).strip(),
        note='Numerical working trees were dirty. The launch/per-run source bundles, not HEAD alone, define the measured code. No unrelated WIP is committed by preservation.'))
    (root / 'KEEP_BASELINE.txt').write_text(
        'RETAIN: canonical classic-fluid improvement baseline, 21–22 October 2026.\n'
        'Do not remove or overwrite during routine experiment/cache cleanup.\n'
        'Keep raw failures, source snapshots, endpoint-artifact flags and all videos/data.\n'
        'Future improvements must use a new dated directory/version, not replace this one.\n'
        'Open comparison.html and ranking/case_ranking.pdf; source docs: docs/fluid_matrix/README.md.\n'
        f'Frozen archive: {archive}\n'
        'This is a local archive; parent warpSPHDevelopment is designated for manual cloud backup.\n')
    (keep / 'RESTORE.txt').write_text(
        'RESTORE THE CLASSIC-FLUID BASELINE\n\n'
        '1. Verify the archive: sha256sum -c <archive>.sha256 (from its directory).\n'
        '2. Extract into an empty directory: tar --zstd -xf <archive>.\n'
        '3. Open <extracted-root>/comparison.html; all HTML data/video links are relative.\n'
        '   The frozen PDF/SVG/PNG ranking and CSV/JSON definitions are in ranking/.\n'
        '4. Verify every payload file using preserve_fluidMatrix.py --verify with --inventory.\n'
        '   The verifier requires Python and zstd, with no GPU or daemon.\n'
        '   No simulations are needed to inspect the evidence.\n'
        '5. For analysis regeneration, use preservation/repository/scripts and the copied\n'
        '   experimental reference; pass --root <extracted-root> explicitly. Runtime versions\n'
        '   are in runtime.json. Source Python snapshots for the measured frontend/core/\n'
        '   integrators/boundaries/plotting are in sources.tar.gz and per-run source bundles.\n'
        '   These snapshots are provenance, not a standalone binary wheel/environment.\n\n'
        'The rounding-sized terminal-step defect is flagged in final_audit.json and the\n'
        'curated report. Keep the original raw endpoints as well as derived corrections.\n')
    excluded = {'preservation/inventory.json', 'preservation/archive_receipt.json'}
    rows = []
    files = sorted(p for p in root.rglob('*') if p.is_file() and str(p.relative_to(root)) not in excluded)
    for i, path in enumerate(files):
        if path.is_symlink():
            raise ValueError(f'Symlinks must be resolved deliberately before archival: {path}')
        rows.append(dict(path=str(path.relative_to(root)), size=path.stat().st_size, sha256=digest(path)))
        if (i + 1) % 1000 == 0:
            print(f'HASHED {i + 1}/{len(files)} files', flush=True)
    inventory = dict(schema=1, baseline='classic-fluid-matrix-2026-10-21-v1',
                     root_name=root.name, created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                     files=rows, total_bytes=sum(r['size'] for r in rows),
                     excludes=sorted(excluded), retention='Keep permanently as baseline; future results use distinct names.')
    write_json(keep / 'inventory.json', inventory)
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(archive.suffix + '.partial')
    with temporary.open('xb') as output:
        compressor = subprocess.Popen(['zstd', '-q', '-3', '-T4', '-c'], stdin=subprocess.PIPE, stdout=output)
        try:
            with tarfile.open(fileobj=compressor.stdin, mode='w|') as tar:
                for path in [*files, keep / 'inventory.json']:
                    tar.add(path, arcname=str(Path(root.name) / path.relative_to(root)), recursive=False)
            compressor.stdin.close()
            if compressor.wait() != 0:
                raise RuntimeError('Archive compression failed')
        finally:
            if compressor.poll() is None:
                compressor.terminate()
        output.flush()
        os.fsync(output.fileno())
    print('ARCHIVE WRITTEN; verifying decompressed member hashes', flush=True)
    checked = verify(temporary, inventory)
    checksum = digest(temporary)
    temporary.rename(archive)
    sidecar = archive.with_name(archive.name + '.sha256')
    sidecar.write_text(f'{checksum}  {archive.name}\n')
    inventory_sidecar = archive.with_name(archive.name + '.inventory.json')
    shutil.copy2(keep / 'inventory.json', inventory_sidecar)
    receipt = dict(schema=1, archive=str(archive), sha256=checksum, archive_bytes=archive.stat().st_size,
                   inventory=str(inventory_sidecar), **checked, verified_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   backup_scope='Local filesystem only. No upload/synchronization performed.')
    write_json(archive.with_name(archive.name + '.verification.json'), receipt)
    write_json(keep / 'archive_receipt.json', receipt)
    write_json(REPO / 'docs/fluid_matrix/evidence/archive_2026-10-22.json', receipt)
    for path in (archive, sidecar, inventory_sidecar, archive.with_name(archive.name + '.verification.json')):
        path.chmod(0o444)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--inventory', type=Path)
    args = parser.parse_args()
    if args.verify:
        path = args.inventory or args.archive.with_name(args.archive.name + '.inventory.json')
        print(json.dumps(verify(args.archive, json.loads(path.read_text())), indent=2))
    else:
        preserve(args.root, args.archive)
