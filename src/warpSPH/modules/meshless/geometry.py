"""Core geometry builds and a version-checked, state-owned timestep handover.

Clone lineage is registered only where MeshlessSystem explicitly clones its state.
A changed source, clone, domain tensor, kernel or conditioning setting invalidates
reuse without reading GPU values. External callers retain the legacy one-slot API.
"""
from dataclasses import dataclass
import torch
from warpSPHCore.mfm import MeshlessWarp
from ...configurations.compute import neighborSearchOptions


@dataclass
class NeighborCache:
    signature: tuple = None
    adjacency: object = None


def cachedAdjacency(positions, supports, domain, schemeConfig, *, owner=None, storeI=True):
    """Exact current neighbors filtered from a validity-checked, padded list.

    The skin changes search work only. Every geometry recomputes moments and
    filters pairs at the current supports. Baseline search remains the default.
    """
    skin = schemeConfig.neighborSkin
    if skin == 1. or owner is None:
        return None
    from warpSPHCore.dataTypes import ParticleState
    from warpSPHCore.enumTypes import SupportScheme
    from warpSPHCore.radiusSearch.verlet import buildVerletList, filterVerletList
    cache = getattr(owner, '_meshlessNeighbors', None)
    if cache is None:
        cache = owner._meshlessNeighbors = NeighborCache()
    signature = (id(domain), domain.dim, _versions(domain.min, domain.max, domain.periodic),
                 positions.shape, positions.device, positions.dtype, skin, storeI)
    prior = cache.adjacency if cache.signature == signature else None
    particles = ParticleState(positions=positions, supports=supports,
        masses=torch.ones_like(supports), kinds=torch.zeros_like(supports, dtype=torch.int32))
    padded = buildVerletList(particles, domain, verletScale=skin,
        supportMode=SupportScheme.SuperSymmetric, priorNeighborhood=prior, storeI=storeI)
    if padded is not prior:
        # A baseline must survive later in-place changes to the public state.
        padded.queryPositions = padded.referencePositions = positions.clone()
        padded.querySupports = padded.referenceSupports = (supports*skin).clone()
        padded.hashMap = None
    cache.signature, cache.adjacency = signature, padded
    return filterVerletList(particles, domain, padded, supportMode=SupportScheme.SuperSymmetric)


def _versions(*values):
    return tuple((id(t), t._version) if isinstance(t, torch.Tensor) else t for t in values)


def _settings(domain, kernel, config):
    return (id(domain), domain.dim, _versions(domain.min, domain.max, domain.periodic), kernel,
            config.condMax, config.faceCondMax)


@dataclass
class GeometryTicket:
    positions: torch.Tensor
    supports: torch.Tensor
    sourceVersions: tuple
    settings: tuple
    geometry: object
    targetVersions: tuple

    def validSource(self):
        return self.sourceVersions == _versions(self.positions, self.supports)


_pendingGeometry = None   # compatibility for direct timestep/step callers


def stashGeometry(positions, supports, domain, kernel, schemeConfig, geometry, *, owner=None):
    if owner is not None and not hasattr(owner, '_meshlessNeighbors'):
        owner._meshlessNeighbors = NeighborCache()
    versions = _versions(positions, supports)
    ticket = GeometryTicket(positions, supports, versions, _settings(domain, kernel, schemeConfig), geometry, versions)
    if owner is not None:
        owner._meshlessGeometry = ticket
    else:
        global _pendingGeometry
        _pendingGeometry = ticket


def inheritGeometry(source, target):
    """Register an exact clone; later in-place writes invalidate its ticket."""
    if hasattr(source, '_meshlessNeighbors'):
        target._meshlessNeighbors = source._meshlessNeighbors
    ticket = getattr(source, '_meshlessGeometry', None)
    if ticket is not None and ticket.validSource() and ticket.targetVersions == _versions(source.positions, source.supports):
        target._meshlessGeometry = GeometryTicket(ticket.positions, ticket.supports, ticket.sourceVersions,
            ticket.settings, ticket.geometry, _versions(target.positions, target.supports))


def takeGeometry(positions, supports, domain, kernel, schemeConfig, *, owner=None):
    global _pendingGeometry
    ticket = getattr(owner, '_meshlessGeometry', None) if owner is not None else None
    if owner is not None:
        owner._meshlessGeometry = None
    legacy = owner is None
    if legacy:
        ticket, _pendingGeometry = _pendingGeometry, None
    if ticket is None or not ticket.validSource() or ticket.settings != _settings(domain, kernel, schemeConfig):
        return None
    if ticket.targetVersions == _versions(positions, supports):
        return ticket.geometry
    if legacy and ticket.positions.shape == positions.shape and ticket.supports.shape == supports.shape:
        if torch.equal(ticket.positions, positions) and torch.equal(ticket.supports, supports):
            return ticket.geometry
    return None


def buildGeometry(positions, supports, lam0, config, schemeConfig, domain, *, owner=None):
    return MeshlessWarp.build(
        positions, supports, domain, config.kernel,
        adjacency=cachedAdjacency(positions, supports, domain, schemeConfig, owner=owner,
                                  storeI=neighborSearchOptions(config)['storeI']),
        storeI=neighborSearchOptions(config)['storeI'],
        cond_max=schemeConfig.condMax, closure=schemeConfig.closure,
        closure_power=schemeConfig.closurePower, rtol=None, cg_tol=schemeConfig.cgTol,
        cg_iters=schemeConfig.cgIters, lam0=lam0, guards=True,
        face_cond_max=schemeConfig.faceCondMax, centred_weights=True, area_cap=False,
        closure_solver=schemeConfig.closureSolver,
        mg_min_particles=schemeConfig.mgMinParticles, mg_max_coarse=schemeConfig.mgMaxCoarse,
        closure_cap=schemeConfig.closureCap)


def startOfStepGeometry(state, domain, config, schemeConfig):
    """The start-of-step geometry: the timestep's one at the same positions / supports when it is there (only the closure solve is added),
    else a fresh build, warm-starting the closure solve from the previous step's lam."""
    geom = takeGeometry(state.positions, state.supports, domain, config.kernel, schemeConfig, owner=state)
    if geom is None:
        return buildGeometry(state.positions, state.supports, state.lam, config, schemeConfig, domain, owner=state)
    if schemeConfig.closure == 'project':
        geom.close(closure_power=schemeConfig.closurePower, cg_tol=schemeConfig.cgTol,
                   cg_iters=schemeConfig.cgIters, lam0=state.lam,
                   closure_solver=schemeConfig.closureSolver,
                   mg_min_particles=schemeConfig.mgMinParticles, mg_max_coarse=schemeConfig.mgMaxCoarse,
                   closure_cap=schemeConfig.closureCap)
    return geom


