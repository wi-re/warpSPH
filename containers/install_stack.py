"""Install explicitly supplied source wheels/distributions without replacing Torch."""
import json
import hashlib
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'stack-manifest.json').read_text())
    selected = [item for name, item in manifest['packages'].items()
                if (name == 'warpSPHRemote') == (sys.argv[1] == 'remote')]
    constraints = ['-c', str(root / 'base-constraints.txt'), '-c', str(root / 'constraints.txt')]
    wheels = root / 'wheels'
    wheels.mkdir(exist_ok=True)
    specifications = []
    for item in selected:
        if item['kind'] == 'source':
            # Build one wheel per exact snapshot, not an editable host path.
            before = set(wheels.glob('*.whl'))
            subprocess.run([sys.executable, '-m', 'pip', 'wheel', '--no-deps',
                            '--no-build-isolation', '-w', str(wheels),
                            str(root / item['snapshot'])], check=True)
            built = set(wheels.glob('*.whl')) - before
            if len(built) != 1:
                raise RuntimeError('Expected one new wheel per source package.')
            specifications.append(str(built.pop()))
        else:
            specifications.append(item['requirement'])
    # A custom base may already contain these release numbers with different
    # source. Force the explicitly supplied application wheels first, without
    # reinstalling Torch or other dependencies. Then resolve dependencies for
    # all paired packages together under the base constraints.
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--force-reinstall',
                    '--no-deps', *constraints, *specifications], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', *constraints,
                    *specifications], check=True)
    (root / 'wheel-hashes.json').write_text(json.dumps({
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(wheels.glob('*.whl'))}, indent=2) + '\n')


if __name__ == '__main__':
    main()
