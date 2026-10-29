# Neighbor-search compute choices

`SimulationConfig.compute` selects how neighbor lists are built independently of the physical scheme:

```python
config, schemeConfig = buildConfig(compute={
    'neighborOmitI': True,  # bare buildConfig has no scheme family; choose explicitly
    'neighborCollection': 'auto',
    'neighborQueryOrder': 'auto',
    'neighborGridBuild': 'scheme',
})
```

| Option | Choices and default |
|---|---|
| `neighborOmitI` | `None`: compressible runner contexts omit COO row indices; other families retain them pending a consumer audit. `True` omits them and `False` restores stored COO. Bare low-level configs retain COO when None. Requires the paired core `storeI` API. |
| `neighborCollection` | `auto` (default): split I/J on CUDA, fused on CPU. `split` and `fused` explicitly select either implementation. |
| `neighborQueryOrder` | `auto` (default): use the existing grid permutation for aliased self queries, input order for independent query/reference layouts. `input` forces input execution order; `spatial` requires aliased positions. |
| `neighborGridBuild` | `scheme` (default): use the scheme's existing pair-grid factory when available. `eager` bypasses that factory. It does not change support-solve/direct-grid operator traversal or add cached-grid support to other schemes. |

The settings round-trip through config files and reach initial sampling, adaptive support, shifts, stage pair lists and replay-triggered rebuilds. Explicit `fused` + `input` recovers the original traversal/collection policy. Changes affect the next rebuild; existing valid lists can still be reused.

Spatial scheduling sorts no particle state. Sorted query slot `s` uses original row `sortIndex[s]`; counts, offsets, I/J and trajectory arrays retain original particle order. The already wrapped/sorted reference positions serve as query positions. Query supports retain their own original-row indexing, even when different from reference supports. Independent sets of equal size are not treated as identical.

The core and frontend neighbor kernels use the configured particle-ID and CSR-offset types. Import-time integer-width selection remains a process configuration, separately from these per-run compute choices.

Measurements and timing/interaction-rate figures are in `warpSPHCore/docs/development/query-order-validation.md`. On the measured GPU, dense shuffled 3D at 2.1M particles improves from 139.7 to 38.5 ms over input-order split collection; a dense regular 2D case regresses about 2%, so input order remains available.

Actual frontend simulations are qualified in [the 2026-10-15 simulation report](microbench/frontend_search_2026-10-15.md): 90 trials / 18,000 steps, including default output pipelining. Core-policy changes preserve states bitwise; larger eager cases use 20–23% less peak Torch memory, with small full-step timing changes. Spatial scheduling is about 1% slower in the measured ordered 3D case; input scheduling remains selectable. Changing grid construction has separately quantified float32 differences.

[Developed sloshing](microbench/sloshing_motion_2026-10-15.md) confirms that motion changes rebuild frequency and locality: 122 / 118 CSR calls per 1,000 impact-window steps versus zero at startup. At 10.7k particles the new defaults improve complete rebuilds 7–8% and the whole loop about 2% against the wide control, with bitwise parity; larger moving cases need separate measurements.

## CSR-only allocation (2026-10-29)

The paired core search option is `storeI=False`: no edge-sized I tensor is
allocated, and the J-only collector skips the separate row-fill launch. CSR
counts, offsets, J ordering, capacity headroom, Verlet validity, support solves
and physical stage evaluations are unchanged. With I omitted, collection uses
the J-only kernel even if `neighborCollection='fused'` was requested. Core
callers keep the historical `storeI=True` default; filters inherit their input
representation unless explicitly overridden. Changing representation rebuilds
rather than overwriting an incompatible in-place allocation.

`AdjacencyList.paddedI` is None on this route. Pair-array shapes come from
`paddedJ`; ordinary kernels already identify I from the CSR row. Explicit
`cooRows()` and `livePairs()` compatibility calls reconstruct temporary rows
without storing them back on the list. Wall fractions, experimental energy
controls, Torch differentiation fallbacks and raw-pair diagnostics may therefore
allocate temporary I; savings in those configurations need their own measurement.
Graph clone/address keys accept the absent tensor. Driven checkpoints omit it,
load old COO files, and can restore either representation.

WCSPH, ACSPH, incompressible and wave defaults remain COO. Their shifting,
boundary, development and diagnostic consumers still need a full allocation and
performance audit before enabling omission by default.

One independent capacity blocker remains: `radiusSearch/csr.py` rejects flat
arrays longer than 2**31-1, including padding. Selecting int64 offsets alone
**does not** remove this Warp array-dimension limit. With the measured B7/n_h4
buffered capacity, this is reached around156³; the earlier176³–192³ projections
were VRAM estimates, not currently executable sizes. Larger cases need chunked
arrays or another backend addressing change as well as sufficient VRAM.
