#!/usr/bin/env python3
"""Verify that every import in the repo still resolves.

Two independent passes:

1. **Runtime pass** -- import every module under ``warpSPH`` for real. This is
   the only way to catch errors that hide in module bodies (bad re-exports,
   circular imports, missing dependencies).
2. **Static pass** -- AST-scan every ``.py`` file and every ``.ipynb`` code
   cell for imports of first-party packages, then check that the module
   resolves and that each imported *symbol* actually exists. This catches
   function-level ("lazy") imports and notebook imports that the runtime pass
   never executes.

Usage::

    python scripts/check_imports.py            # both passes
    python scripts/check_imports.py --runtime  # runtime pass only (fast-ish)
    python scripts/check_imports.py --static   # static pass only (no GPU needed)
    python scripts/check_imports.py -v         # list every module as it loads

Exit code is 0 only when both passes are clean.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import json
import pkgutil
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"

# Everything we own is named warpSPH*: the package itself, the sibling repos
# (warpSPHCore/Plotting/Integrators), the top-level bootstrap/run modules, and
# the warpSPHCore_config shim the notebooks import. Anything else is
# third-party and not this script's problem.
FIRST_PARTY_PREFIX = "warpSPH"
# optional first-party packages: their imports are verified only when the package is installed (`pip install -e "warpSPH/[boundaries]"`)
OPTIONAL_PACKAGES = {"warpSPHBoundaries"}

SKIP_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints", "build", "dist", ".venv", "venv"}


@dataclass
class Failure:
    where: str
    what: str
    detail: str

    def __str__(self) -> str:
        return f"  {self.where}\n      {self.what}\n      -> {self.detail}"


def is_first_party(module: str) -> bool:
    return module.startswith(FIRST_PARTY_PREFIX)


# --------------------------------------------------------------------------
# Pass 1: runtime import of the whole package
# --------------------------------------------------------------------------
def runtime_pass(verbose: bool) -> list[Failure]:
    failures: list[Failure] = []
    try:
        import warpSPH
    except Exception as exc:  # pragma: no cover - catastrophic
        return [Failure("import warpSPH", type(exc).__name__, str(exc))]

    names = sorted(m.name for m in pkgutil.walk_packages(warpSPH.__path__, "warpSPH."))
    print(f"[runtime] importing {len(names)} modules under warpSPH ...")
    for name in names:
        if verbose:
            print(f"          {name}")
        try:
            importlib.import_module(name)
        except Exception as exc:
            tb = traceback.format_exc(limit=6).strip().splitlines()
            failures.append(Failure(name, f"{type(exc).__name__}: {exc}", tb[-1] if tb else ""))
    return failures


# --------------------------------------------------------------------------
# Pass 2: static scan of every .py / .ipynb in the repo
# --------------------------------------------------------------------------
def iter_source_files() -> list[Path]:
    files: list[Path] = []
    for path in REPO.rglob("*"):
        if path.suffix not in (".py", ".ipynb"):
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def notebook_source(path: Path) -> str:
    """Concatenate a notebook's code cells into something ast can parse."""
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return ""
    lines: list[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        # nbformat permits `source` to be one string as well as a list of
        # lines. Iterating a string walks it character by character, which
        # turns every cell it appears in into nonsense the parser rejects.
        if isinstance(source, str):
            source = source.splitlines(keepends=True)
        for line in source:
            stripped = line.lstrip()
            # IPython magics / shell escapes are not valid Python.
            if stripped.startswith(("%", "!", "?")):
                line = "pass\n"
            lines.append(line.rstrip("\n"))
        lines.append("")
    return "\n".join(lines)


def module_of_file(path: Path) -> str | None:
    """Dotted package name for a file under src/, else None."""
    try:
        rel = path.relative_to(SRC)
    except ValueError:
        return None
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][: -len(".py")]
    return ".".join(parts)


def resolve_relative(module: str | None, level: int, name: str | None, is_pkg: bool) -> str | None:
    """Turn a relative import into an absolute dotted name."""
    if module is None:
        return None
    parts = module.split(".")
    if not is_pkg:
        parts = parts[:-1]
    # level 1 == current package; each extra level walks up one more.
    if level - 1 > len(parts):
        return None
    base = parts[: len(parts) - (level - 1)]
    return ".".join(base + ([name] if name else []))


def collect_imports(tree: ast.AST, owner: str | None, is_pkg: bool):
    """Yield (module, symbols, lineno) for every first-party import."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if is_first_party(alias.name):
                    yield alias.name, [], node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                module = resolve_relative(owner, node.level, node.module, is_pkg)
                if module is None:
                    continue
            else:
                module = node.module or ""
            if not is_first_party(module):
                continue
            symbols = [a.name for a in node.names if a.name != "*"]
            yield module, symbols, node.lineno


def static_pass(verbose: bool) -> list[Failure]:
    failures: list[Failure] = []
    files = iter_source_files()
    print(f"[static]  scanning {len(files)} source files ...")

    module_cache: dict[str, object | None] = {}

    def load(module: str):
        if module not in module_cache:
            try:
                module_cache[module] = importlib.import_module(module)
            except Exception:
                module_cache[module] = None
        return module_cache[module]

    checked = 0
    skippedOptional = 0
    for path in files:
        rel = path.relative_to(REPO)
        source = notebook_source(path) if path.suffix == ".ipynb" else path.read_text(
            encoding="utf-8", errors="replace"
        )
        if not source.strip():
            continue
        try:
            tree = ast.parse(source, filename=str(rel))
        except SyntaxError as exc:
            failures.append(Failure(f"{rel}:{exc.lineno}", "SyntaxError", str(exc.msg)))
            continue

        owner = module_of_file(path)
        is_pkg = path.name == "__init__.py"
        for module, symbols, lineno in collect_imports(tree, owner, is_pkg):
            checked += 1
            if verbose:
                print(f"          {rel}:{lineno} {module} {symbols or ''}")
            try:
                spec = importlib.util.find_spec(module)
            except (ImportError, AttributeError, ValueError) as exc:
                spec = None
                if verbose:
                    print(f"            find_spec raised: {exc}")
            if spec is None and module.split(".")[0] in OPTIONAL_PACKAGES:
                skippedOptional += 1
                continue
            if spec is None:
                failures.append(
                    Failure(f"{rel}:{lineno}", f"no module named {module!r}", "module not found")
                )
                continue
            if not symbols:
                continue
            mod = load(module)
            if mod is None:
                failures.append(
                    Failure(f"{rel}:{lineno}", f"{module} failed to import", "see runtime pass")
                )
                continue
            for symbol in symbols:
                if hasattr(mod, symbol):
                    continue
                # `from pkg import submodule` is legal even without a re-export.
                if importlib.util.find_spec(f"{module}.{symbol}") is not None:
                    continue
                failures.append(
                    Failure(f"{rel}:{lineno}", f"cannot import {symbol!r} from {module}", "symbol missing")
                )
    print(f"[static]  checked {checked} first-party imports")
    if skippedOptional:
        print(f"[static]  skipped {skippedOptional} imports of optional packages that are not installed ({', '.join(sorted(OPTIONAL_PACKAGES))})")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runtime", action="store_true", help="run only the runtime import pass")
    parser.add_argument("--static", action="store_true", help="run only the static AST pass")
    parser.add_argument("-v", "--verbose", action="store_true", help="print each module/import as it is checked")
    args = parser.parse_args()

    run_runtime = args.runtime or not args.static
    run_static = args.static or not args.runtime

    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

    failures: list[Failure] = []
    if run_runtime:
        failures += runtime_pass(args.verbose)
    if run_static:
        failures += static_pass(args.verbose)

    print()
    if failures:
        print(f"FAILED -- {len(failures)} problem(s):\n")
        for failure in failures:
            print(failure)
        return 1
    print("OK -- all imports resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
