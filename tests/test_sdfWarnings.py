"""The SDF module must not install a process-wide warning filter, and its own
functions must not emit warnings (the only one it did -- `.T` on a 1-D
tensor in sdHorseshoe -- was the reason for a blanket
warnings.filterwarnings("ignore") that silenced every warning in any
process importing it)."""

import warnings

import torch


def test_import_installs_no_blanket_ignore():
    before = list(warnings.filters)
    import importlib
    import warpSPH.geometry.sdfFunctionality.implicitFunctions as m
    importlib.reload(m)
    added = [f for f in warnings.filters if f not in before]
    assert not any(f[0] == 'ignore' and f[2] is Warning for f in added), added


def test_sdf_functions_emit_no_warnings():
    from warpSPH.geometry.sdf import getSDF
    from warpSPH.geometry.sdfFunctionality.implicitFunctions import functionDict
    p = (torch.rand(32, 2, dtype=torch.float64) - 0.5) * 3
    # star / ring are broken independently (np.cos / np.array given device=)
    for name in (n for n in functionDict if n not in ('star', 'ring')):
        d = getSDF(name)
        args = [a.to(p.dtype) if torch.is_tensor(a) else a for a in d['sample']]
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            pp = p.clone().requires_grad_(True)
            d['function'](pp, *args).sum().backward()
        assert not rec, (name, [str(r.message)[:80] for r in rec])
