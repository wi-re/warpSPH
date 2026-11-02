"""Owned search envelopes and complete-attempt renewal for fixed-layout SPH.

Candidate radii are search data, never physical smoothing supports. If each
anchor radius covers h_i + 2 max_j |x_j - anchor_j|, the triangle inequality
covers every physical super-symmetric pair, including periodic images. Support
solvers pass their peak iterates through the same check before publishing h.
"""
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from warpSPHCore import SupportScheme, buildVerletList
from warpSPHIntegrators.state import ResourceInvalidated

from ..configurations.compute import neighborSearchOptions


def _motion(positions, anchors, domain):
    delta = positions - anchors
    extent = domain.max - domain.min
    wrapped = torch.remainder(delta + extent / 2, extent) - extent / 2
    delta = torch.where(domain.periodic.reshape(1, -1), wrapped, delta)
    return 2 * torch.linalg.vector_norm(delta, dim=-1).amax()


@dataclass(frozen=True, eq=False)
class CoverageEnvelope:
    """Owned UID-keyed reach at fixed anchor positions for one stage path."""
    stage_path: tuple
    input_ids: torch.Tensor
    anchor_positions: torch.Tensor
    radii: torch.Tensor

    def build(self, config):
        # The core builder scales supports; only this synthetic search carrier
        # is divided by that scale. Physical state/configuration are untouched.
        search = SimpleNamespace(positions=self.anchor_positions,
                                 supports=self.radii / config.verletScale)
        return buildVerletList(search, config.domain, verletScale=config.verletScale,
            supportMode=SupportScheme.SuperSymmetric, **neighborSearchOptions(config))


@dataclass(frozen=True, eq=False)
class CoverageRenewal:
    envelope: CoverageEnvelope
    purpose: str
    required_supports: torch.Tensor
    motion_budget: torch.Tensor
    previous_radii: torch.Tensor


class CoverageInvalidated(ResourceInvalidated):
    def __init__(self, renewal):
        self.renewal = renewal
        super().__init__(f'neighbor coverage changed in stage {renewal.envelope.stage_path} '
                         f'({renewal.purpose}); renew coverage and retry the complete solve')


class CoverageRetryExhausted(ResourceInvalidated):
    def __init__(self, failure, attempts):
        self.renewal, self.attempts = failure.renewal, attempts
        super().__init__(f'coverage retry budget exhausted after {attempts} complete attempts; '
                         f'stage {self.renewal.envelope.stage_path}')


class CoverageBuilder:
    """One RHS's immutable candidate, checked also against support-solver peaks.

    Without a pinned candidate, admit a native list at the incoming geometry.
    Then keep it through all support iterates; renewal unwinds the whole attempt.
    Never replace a pinned list or return equation values after a failed check.
    """
    def __init__(self, adjacency, request, ids):
        self.adjacency, self.request, self.ids = adjacency, request, ids

    def __call__(self, particles, domain, **options):
        fixed = self.adjacency
        if fixed is None:
            self.adjacency = buildVerletList(particles, domain, **options)
            return self.adjacency
        positions, supports = particles.positions, particles.supports
        if positions.shape != fixed.queryPositions.shape or positions.shape != fixed.referencePositions.shape:
            raise ValueError('coverage renewal cannot change the particle population')
        if fixed.queryPositions is not fixed.referencePositions and not torch.equal(fixed.queryPositions, fixed.referencePositions):
            raise ValueError('coverage renewal requires a self-particle candidate')
        motion = _motion(positions, fixed.queryPositions, domain)
        needed = supports + motion
        # Both reach arrays are checked, independent of aliasing or N == E.
        if bool(((needed <= fixed.querySupports) & (needed <= fixed.referenceSupports) & (supports > 0)).all()):
            return fixed
        if not bool((torch.isfinite(needed) & (supports > 0)).all()):
            raise ValueError('coverage requires finite geometry and positive finite supports')
        previous = torch.maximum(fixed.querySupports, fixed.referenceSupports)
        envelope = CoverageEnvelope(self.request.stage_path, self.ids.detach().clone(),
            fixed.queryPositions.detach().clone(),
            (torch.maximum(previous, needed) * 1.25).detach().clone())
        raise CoverageInvalidated(CoverageRenewal(envelope, self.request.purpose,
            supports.detach().clone(), motion.detach().clone(), previous.detach().clone()))
