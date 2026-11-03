"""Accepted-boundary permutations of replicated particles and owned row chunks.

The numerical attempt has one fixed particle basis. A changed basis builds only
the new owner's J buffers; old chunk-local edge work is never gathered or reused.
"""
from warpSPHCore.radiusSearch import buildChunkedVerletList, chunkedVerletStaleFlag_

from .nativeLayout import NativeMortonLayout


class ChunkedMortonLayout(NativeMortonLayout):
    def __init__(self, system, integrator, scheme, *, retain_initial_order=False):
        super().__init__(system, integrator, scheme, retain_initial_order=retain_initial_order)
        from ..modules.meshless.sources import ConservedSources
        from ..modules.forcing import FourierOU
        source = getattr(system, 'sourceTerms', None)
        if source is not None and (type(source) is not ConservedSources or type(source.acceleration) is not FourierOU):
            raise ValueError('chunked Morton sources require the audited ConservedSources/FourierOU adapter')
        self.communication = system.adjacency.communication

    @staticmethod
    def _stale(system, config):
        s, a = system.state, system.adjacency
        return chunkedVerletStaleFlag_(s.positions, s.positions, s.supports, s.supports,
            config.domain, a, mode=a.mode, verletScale=config.verletScale).reshape(())

    def prepare(self, system, config, scheme):
        old = system.adjacency
        comm = self.communication
        local = (lambda tag, fn: fn()) if comm is None else comm.localPhase
        if old.communication is not comm:
            raise ValueError('chunked Morton communicator changed outside a restart')
        if self.ids is not None:
            def validity():
                self._check_ids(system)
                return bool(self._stale(system, config))
            stale = local('layout:validity', validity)
            if comm is not None:
                stale = comm.any(stale, tag='layout:global-stale')
            self.pending = (self._owner(system, config), stale)
        prepared, cfg, sc, changed = local('layout:prepare',
            lambda: super(ChunkedMortonLayout, self).prepare(system, config, scheme))
        if comm is not None:
            comm.agree('layout:transition', (changed, self.generation))
            comm.assertReplicated('layout:UIDs', dict(UIDs=prepared.state.UIDs), exact=True)
        if changed and prepared.adjacency is None:
            # A forced rebuild advances the generation and repartitions whole
            # rows. No flat J, per-edge permutation or stale W is constructed.
            prepared.adjacency = buildChunkedVerletList(prepared.state, cfg.domain,
                policy=old.plan.policy, supportMode=old.mode, verletScale=cfg.verletScale,
                priorNeighborhood=old, communication=comm, storeI=old.storeI,
                forceRebuild=True, orderingKey=f'morton:{self.generation}')
        return prepared, cfg, sc, changed
