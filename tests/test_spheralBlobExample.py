"""Ensure the bundled-driver adapter honors solver selection without tuning it."""
import ast
import sys
from pathlib import Path
import pytest
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_spheralBlobExample import _blob_Compatibility
import run_spheralBlobExample as driver


DISPATCH='''
if crksph:
    hydro=CRKSPH(dataBase=db)
elif psph:
    hydro=PSPH()
if fsisph:
    hydro=FSISPH()
elif gsph:
    hydro=GSPH()
else:
    hydro=SPH()
'''


def namespace(selected):
    n={name:False for name in ('crksph','psph','fsisph','gsph')}
    if selected:n[selected]=True
    n.update(db=object(),WT=object())
    for name in ('CRKSPH','PSPH','FSISPH','GSPH','SPH'):
        n[name]=lambda _name=name,**kw:(_name,kw)
    return n


def test_upstream_two_independent_branches_overwrite_crksph():
    n=namespace('crksph');exec(DISPATCH,n)
    assert n['hydro'][0]=='SPH'


@pytest.mark.parametrize('flag,expected',[('crksph','CRKSPH'),('psph','PSPH'),('fsisph','FSISPH'),('gsph','GSPH'),(None,'SPH')])
def test_adapter_honors_exclusive_selection_and_preserves_defaults(flag,expected):
    tree=_blob_Compatibility().visit(ast.parse(DISPATCH));ast.fix_missing_locations(tree)
    n=namespace(flag);exec(compile(tree,'<dispatch>','exec'),n)
    assert n['hydro'][0]==expected
    if flag=='crksph':
        assert n['hydro'][1]['dataBase'] is n['db'] and n['hydro'][1]['W'] is n['WT']


def test_paper_adapter_explicitly_enrolls_extent_and_volume(monkeypatch):
    monkeypatch.setattr(driver,'_blob_args',SimpleNamespace(paper=True),raising=False)
    tree=_blob_Compatibility().visit(ast.parse('nodes=nodeListConstructor()\ncontroller=SpheralController()'))
    calls=[node.value for node in tree.body]
    assert ast.literal_eval(calls[0].keywords[0].value)==4.
    assert calls[1].keywords[0].arg=='volumeType'
    assert calls[1].keywords[0].value.id=='RKSumVolume'


def test_recorded_outputs_replace_only_obsolete_control_writes():
    tree=_blob_Compatibility().visit(ast.parse('control.updateViz(1)\ncontrol.dropRestartFile()\ncontrol.advance(2)\nother.updateViz(3)'))
    assert [node.value.func.attr for node in tree.body]==['advance','updateViz']


def test_independent_paper_spline_is_normalized_and_compact():
    import numpy as np
    from scipy.integrate import quad
    from audit_spheralPaperPolicies import w7
    assert abs(quad(lambda q:2*np.pi*q*w7(q),0,4,epsabs=1e-11,points=[1,2,3])[0]-1)<1e-10
    assert np.all(w7(np.array([4.,5.]))==0)
    assert np.all(w7(np.linspace(0,4,100))>=0)
