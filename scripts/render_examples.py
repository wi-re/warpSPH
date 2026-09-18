#!/usr/bin/env python3
"""Re-run every example at its shipped settings and refresh its exports.

Each example under ``examples/`` is a thin ``caseMain`` wrapper carrying the
settings that example ships with (its ``PRESET``). This runs those wrappers --
not the bare cases -- so what comes out is what the example claims to produce,
and files the results next to the notebook that references them:

    examples/<family>/outputs/<name>.gif   the animation the notebook embeds
    examples/<family>/outputs/<name>.mp4   the same, as video
    examples/<family>/outputs/<name>.png   the final frame

``<name>`` is read from the sibling notebook's ``![](outputs/<name>.gif)``
reference, so the file that lands is the file the notebook asks for; if there
is no notebook (or no reference) the wrapper's own filename is used.

Usage::

    scripts/render_examples.py                     # everything (hours -- see --list)
    scripts/render_examples.py --list              # what would run, and where it lands
    scripts/render_examples.py --only impact ldc   # substring match on the wrapper path
    scripts/render_examples.py --skip openFlow
    scripts/render_examples.py --dry-run
    scripts/render_examples.py --trace             # also write a trajectory.h5 per run
    scripts/render_examples.py --only sod -- --nx 64 --tLimit 0.1    # forwarded flags

``--trace`` adds ``--store --storeMode trajectory``, which writes every
particle's state every ``exportInterval`` of simulated time to a single
``trajectory.h5`` -- the per-particle traces the datagen pipeline consumes.
Those stay in the export tree (they are large and are not example artefacts)
and their paths are printed in the summary.

These are **full-length runs at the shipped resolution**, not smoke tests: a
long one (the lid-driven cavity is 60k steps and 6k frames) takes tens of
minutes. Use ``--only`` unless you mean to re-render everything, and note that
``scripts/run_sweep.py`` is the right tool for "does every case still run".
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = REPO / "examples"

#: `![](outputs/<name>.gif)` in a notebook's intro cell.
GIF_REFERENCE = re.compile(r"!\[[^\]]*\]\(outputs/([^)]+)\.gif\)")


@dataclass
class Example:
    """One runnable example wrapper and where its artefacts belong."""

    script: Path                     # examples/<family>/[<dir>/]<stem>.py
    name: str                        # basename of the artefacts, no extension
    notebook: Path | None = None

    @property
    def outputs(self) -> Path:
        return self.script.parent / "outputs"

    def publishDir(self, publishRoot: Path | None) -> Path:
        """Where this example's gif/mp4/png get copied to.

        `publishRoot` given (`--publishRoot`) -> `<publishRoot>/<name>/`,
        `name` already being the notebook-derived, gallery-wide-unique
        artefact name `outputs()` itself keys files by, so this needs no
        further per-family nesting. `None` (default) -> the git-tracked
        `outputs()` next to the notebook, i.e. today's always-on "refresh the
        shipped docs" behavior, unchanged for a deliberate, manual re-render.
        """
        if publishRoot is None:
            return self.outputs
        return Path(publishRoot) / self.name

    @property
    def label(self) -> str:
        return str(self.script.relative_to(EXAMPLES))


@dataclass
class Result:
    example: str
    status: str                      # ok | fail | timeout | no-output
    seconds: float
    artefacts: list[str] = field(default_factory=list)
    trajectory: str | None = None
    logPath: str = ""


def discover() -> list[Example]:
    """Every `caseMain` wrapper under `examples/`, with its artefact name."""
    found: list[Example] = []
    for script in sorted(EXAMPLES.rglob("*.py")):
        text = script.read_text(encoding="utf-8", errors="replace")
        # The wrappers are exactly the files that hand a case to the runner;
        # `utils.py`-style shims and helper modules do not.
        if "caseMain(" not in text:
            continue
        notebook = siblingNotebook(script)
        found.append(Example(script, artefactName(script, notebook), notebook))
    return found


def siblingNotebook(script: Path) -> Path | None:
    """The notebook this wrapper belongs to, if there is one.

    Wrapper and notebook stems now agree, so the exact (case-insensitive) match
    is the normal path. The slot-number fallback (`08-`) is kept because it is
    what makes a *newly added* example work before anyone has thought about
    naming, and because notebooks and wrappers have drifted apart here before.
    """
    notebooks = sorted(script.parent.glob("*.ipynb"))
    stem = script.stem.lower()
    for notebook in notebooks:
        if notebook.stem.lower() == stem:
            return notebook
    slot = re.match(r"(\d+)-", script.stem)
    if slot:
        matches = [n for n in notebooks if n.stem.startswith(f"{slot.group(1)}-")]
        if len(matches) == 1:
            return matches[0]
    return None


def artefactName(script: Path, notebook: Path | None) -> str:
    """What the notebook calls the animation, else the wrapper's own name."""
    if notebook is not None:
        match = GIF_REFERENCE.search(notebook.read_text(encoding="utf-8", errors="replace"))
        if match:
            return match.group(1)
    return script.stem


def run(example: Example, root: Path, args, passthrough: list[str]) -> Result:
    logPath = root / "logs" / f"{example.name}.log"
    logPath.parent.mkdir(parents=True, exist_ok=True)
    exportRoot = root / "export" / example.name

    cmd = [sys.executable, str(example.script),
           "--exportRoot", str(exportRoot),
           "--caseName", example.name,
           "--plot", "--no-show", "--video"]
    if args.trace:
        cmd += ["--store", "--storeMode", "trajectory"]
    if not args.verbose:
        cmd += ["--quiet"]
    cmd += passthrough

    if args.dryRun:
        print(f"\n      $ {' '.join(cmd)}")
        return Result(example.label, "ok", 0.0)

    start = time.monotonic()
    with logPath.open("w") as log:
        log.write(f"$ {' '.join(cmd)}\n\n")
        log.flush()
        try:
            proc = subprocess.run(cmd, cwd=REPO, stdout=log,
                                  stderr=subprocess.STDOUT, timeout=args.timeout)
            status = "ok" if proc.returncode == 0 else "fail"
        except subprocess.TimeoutExpired:
            status = "timeout"
            log.write(f"\n\n== killed after {args.timeout}s ==\n")
    seconds = time.monotonic() - start

    result = Result(example.label, status, seconds, logPath=display(logPath))
    if status != "ok":
        return result
    return collect(example, exportRoot, result, args.publishRoot)


def collect(example: Example, exportRoot: Path, result: Result,
           publishRoot: Path | None = None) -> Result:
    """Copy the run's gif/mp4/final frame into `example.publishDir(publishRoot)`."""
    runDirs = sorted((p for p in exportRoot.glob("*") if p.is_dir()),
                     key=lambda p: p.stat().st_mtime)
    if not runDirs:
        result.status = "no-output"
        return result
    runDir = runDirs[-1]

    publishDir = example.publishDir(publishRoot)
    publishDir.mkdir(parents=True, exist_ok=True)
    for source, suffix in ((runDir / "out.gif", ".gif"),
                           (runDir / "output.mp4", ".mp4")):
        if source.exists():
            target = publishDir / f"{example.name}{suffix}"
            shutil.copy(source, target)
            result.artefacts.append(display(target))

    frames = sorted((runDir / "images").glob("frame_*.png"))
    if frames:
        target = publishDir / f"{example.name}.png"
        shutil.copy(frames[-1], target)
        result.artefacts.append(display(target))

    trajectory = runDir / "trajectory.h5"
    if trajectory.exists():
        result.trajectory = display(trajectory)

    if not result.artefacts:
        # A run that plotted nothing is a failure of this script's purpose even
        # though the simulation itself exited cleanly -- most often ffmpeg
        # missing, which `encodeFrames` only warns about.
        result.status = "no-output"
    return result


def display(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def tail(path: str, n: int = 12) -> str:
    try:
        lines = (REPO / path).read_text(errors="replace").splitlines()
    except OSError:
        return "      | <no log>"
    return "\n".join(f"      | {line}" for line in lines[-n:])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", metavar="MATCH", default=[],
                        help="only examples whose path contains one of these")
    parser.add_argument("--skip", nargs="+", metavar="MATCH", default=[],
                        help="skip examples whose path contains one of these")
    parser.add_argument("--trace", action="store_true",
                        help="also write a per-particle trajectory.h5 for each run")
    parser.add_argument("--timeout", type=float, default=7200,
                        help="per-example timeout in seconds (default: 7200)")
    parser.add_argument("--outRoot", default=None,
                        help="parent for the run directories (default: <repo>/export/renders)")
    parser.add_argument("--publishRoot", default=None,
                        help="copy each example's gif/mp4/png under <publishRoot>/<name>/ "
                             "instead of the git-tracked examples/<family>/outputs/ next to "
                             "its notebook -- use this for an automated/nightly run so it "
                             "doesn't dirty the shipped docs media on every pass; omit for a "
                             "deliberate 'refresh the shipped docs' run")
    parser.add_argument("--list", action="store_true",
                        help="list the examples, their artefact names, and exit")
    parser.add_argument("--dry-run", dest="dryRun", action="store_true",
                        help="print the command per example without running it")
    parser.add_argument("--verbose", action="store_true",
                        help="do not pass --quiet to the examples")
    parser.add_argument("rest", nargs="*",
                        help="extra flags forwarded to every example (after --)")
    args = parser.parse_args()

    examples = discover()
    if args.only:
        examples = [e for e in examples if any(m in e.label for m in args.only)]
    if args.skip:
        examples = [e for e in examples if not any(m in e.label for m in args.skip)]

    if args.list:
        width = max((len(e.label) for e in examples), default=0)
        for example in examples:
            notebook = "" if example.notebook else "   (no notebook)"
            print(f"  {example.label:<{width}}  ->  "
                  f"{display(example.publishDir(args.publishRoot))}/{example.name}.gif{notebook}")
        return 0

    if not examples:
        print("no examples matched", file=sys.stderr)
        return 2

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root = Path(args.outRoot) if args.outRoot else REPO / "export" / "renders"
    root = root / stamp
    root.mkdir(parents=True, exist_ok=True)

    print(f"== warpSPH example renders -- {len(examples)} examples"
          f"{', with traces' if args.trace else ''} ==")
    print(f"   runs land in: {display(root)}")
    print()

    results: list[Result] = []
    for i, example in enumerate(examples, 1):
        print(f"[{i:>2}/{len(examples)}] {example.label:<44}", end="", flush=True)
        result = run(example, root, args, args.rest)
        results.append(result)
        mark = {"ok": "ok", "fail": "FAIL", "timeout": "TIMEOUT",
                "no-output": "NO OUTPUT"}[result.status]
        print(f" {mark:>9}  {result.seconds:7.1f}s")

    (root / "summary.json").write_text(json.dumps(
        {"timestamp": stamp, "trace": args.trace, "extraArgs": args.rest,
         "results": [r.__dict__ for r in results]}, indent=2))

    failed = [r for r in results if r.status != "ok"]
    print()
    print(f"== {len(results) - len(failed)}/{len(results)} rendered ==")
    for result in results:
        for artefact in result.artefacts:
            print(f"   {artefact}")
        if result.trajectory:
            print(f"   {result.trajectory}   (trace)")
    for result in failed:
        print()
        print(f"  {result.example} [{result.status}] -> {result.logPath}")
        print(tail(result.logPath))
    print()
    print(f"summary: {display(root / 'summary.json')}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
