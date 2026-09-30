"""Shared machinery for the benchmark suites: scheme registry, instrumented
run loop, metrics, and reporting. Suite-specific logic (what to sweep, what
to compare against) lives in the sibling suite packages (`benchmarks/wave/`).
"""

from .schemes import (
    MULTISTEP_SCHEMES,
    SchemeSpec,
    SCHEMES,
    ACCURACY_DEFAULT,
    PERFORMANCE_DEFAULT,
    STABILITY_EXPLICIT_DEFAULT,
    STABILITY_IMPLICIT_DEFAULT,
    getScheme,
    getSchemes,
)
from .runner import RunRecord, RecordingSolver, buildWaveCase, runScheme
from .metrics import relL2, L1, effectiveOrder, loglogFit, fmt
from . import report
from . import scaling

__all__ = [
    'MULTISTEP_SCHEMES', 'SchemeSpec', 'SCHEMES',
    'ACCURACY_DEFAULT', 'PERFORMANCE_DEFAULT',
    'STABILITY_EXPLICIT_DEFAULT', 'STABILITY_IMPLICIT_DEFAULT',
    'getScheme', 'getSchemes',
    'RunRecord', 'RecordingSolver', 'buildWaveCase', 'runScheme',
    'relL2', 'L1', 'effectiveOrder', 'loglogFit', 'fmt',
    'report', 'scaling',
]
