"""Sequential, Remote-visible RM diagnostics campaign with durable progress.

Run the pilot before spending time on qualification resolutions:
    python scripts/study_materialEOS.py --phase pilot
    python scripts/study_materialEOS.py --phase refinement

Every child records video/progress and registers with the resident companion.
A failure or a Stop command in Remote stops the queue. It does not restart the
case or advance to the next one. No detached simulation children are created.
"""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
import re


def studyCases():
    def case(name, nx=64, switch='NoneSwitch', amplitude=.01, duration=2., **kwargs):
        return dict(name=name, nx=nx, switch=switch, amplitude=amplitude, time=duration, **kwargs)
    return {
        'materialDensity': [case('linear64-sameMaterial', densityPolicy='sameMaterial'),
            case('linear64-massReplacement', densityPolicy='massReplacement'),
            case('nonlinear64-sameMaterial', amplitude=.1, duration=6., xLeft=-12, xRight=20, densityPolicy='sameMaterial')],
        'pilot': [case('linear64'), case('linear64-cullen', switch='CullenDehnen2010'),
                  case('nonlinear64', amplitude=.1, duration=6., xLeft=-12, xRight=20),
                  case('nonlinear64-cullen', switch='CullenDehnen2010', amplitude=.1, duration=6., xLeft=-12, xRight=20)],
        'refinement': [case('linear128', nx=128), case('linear256', nx=256),
                       case('linear128-cullen', nx=128, switch='CullenDehnen2010'),
                       case('linear64-halfdt', cfl=.075), case('linear64-padding', xLeft=-6, xRight=8),
                       case('nonlinear128', nx=128, amplitude=.1, duration=6., xLeft=-12, xRight=20),
                       case('nonlinear128-cullen', nx=128, switch='CullenDehnen2010', amplitude=.1, duration=6., xLeft=-12, xRight=20)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=tuple(studyCases()), default='pilot')
    parser.add_argument('--only', nargs='+', help='Run only the named members of this phase.')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    cases = studyCases()[args.phase]
    if args.only:
        unknown = set(args.only) - {c['name'] for c in cases}
        if unknown:
            parser.error(f'Unknown {args.phase} cases: {sorted(unknown)}')
        cases = [c for c in cases if c['name'] in args.only]
    root = args.output or Path('results/materialEOS') / ('study_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    commands = []
    for case in cases:
        command = [sys.executable, 'scripts/probe_materialEOS.py', '--mode', 'rm', '--live',
                   '--name', 'RM study ' + case['name'], '--output', str(root / case['name'])]
        for key, value in case.items():
            if key != 'name':
                command.extend(['--' + key, str(value)])
        commands.append(command)
    if args.dry_run:
        print(json.dumps(commands, indent=2))
        return 0
    root.mkdir(parents=True, exist_ok=False)
    manifest = dict(phase=args.phase, status='running', cases=[dict(settings=c, command=cmd, status='pending')
                    for c, cmd in zip(cases, commands)])
    path = root / 'campaign.json'
    def save():
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(manifest, indent=2) + '\n')
        temporary.replace(path)
    save()
    for case, command, entry in zip(cases, commands, manifest['cases']):
        log = root / (case['name'] + '.log')
        entry.update(status='running', log=str(log), startedAt=time.time())
        save()
        print(f'START {case["name"]}; Remote name: RM study {case["name"]}; log: {log}', flush=True)
        with log.open('w') as output:
            process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    time.sleep(10)
                    with log.open('rb') as recent:
                        recent.seek(max(0, log.stat().st_size-8192))
                        tail = recent.read().decode(errors='replace').replace('\r', '\n').splitlines()
                    progress = [line for line in tail if '| t=' in line or 'Remote run registered:' in line]
                    if progress:
                        last = progress[-1]
                        fields = re.findall(r'(?:^|\|\s*)((?:t|interfaceCosAmplitude|circulationUpper|interactionAlphaMean|relativeEnergyDrift)=[^ |]+)', last)
                        print(case['name'] + ': ' + (' | '.join(fields) if fields else last), flush=True)
                code = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                entry.update(status='interrupted', finishedAt=time.time())
                manifest['status'] = 'interrupted'
                save()
                raise
        entry.update(status='completed' if code == 0 else 'stopped' if code == 130 else 'failed',
                     returncode=code, finishedAt=time.time())
        save()
        print(f'END {case["name"]}: {entry["status"]}', flush=True)
        if code:
            manifest['status'] = entry['status']
            save()
            return code
    manifest['status'] = 'completed'
    save()
    print('Campaign complete: ' + str(path), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
