"""Unit tests for the benchmark error metrics (`benchmarks/common/metrics.py`)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmarks.common.metrics import L1


def test_L1_hand_computed():
    a = torch.tensor([1.0, 2.0, 4.0, 10.0])
    b = torch.tensor([1.5, 2.0, 3.0, 20.0])
    # |a-b| = [0.5, 0, 1, 10] -> mean 2.875
    assert L1(a, b) == 2.875


def test_L1_mask_selects_sample_set():
    a = torch.tensor([1.0, 2.0, 4.0, 10.0])
    b = torch.tensor([1.5, 2.0, 3.0, 20.0])
    mask = torch.tensor([True, True, True, False])
    assert math.isclose(L1(a, b, mask), 0.5, rel_tol=1e-12)   # (0.5+0+1)/3


def test_L1_is_not_normalised():
    a = torch.tensor([1.0, 3.0])
    b = torch.tensor([0.0, 0.0])
    assert L1(a, b) == 2.0
    assert L1(100 * a, 100 * b) == 200.0


def test_L1_nonfinite_and_empty_are_nan():
    ok = torch.ones(3)
    assert math.isnan(L1(torch.tensor([1.0, float('nan'), 1.0]), ok))
    assert math.isnan(L1(ok, ok, torch.zeros(3, dtype=torch.bool)))
