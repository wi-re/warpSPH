"""Static scanner diagnostics must survive missing symbols and bad finders."""
import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.fixture
def scanner(tmp_path,monkeypatch):
    path=Path(__file__).resolve().parents[1]/'scripts/check_imports.py'
    name='_test_import_scanner'
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules,name,module)
    spec.loader.exec_module(module)
    package=tmp_path/'warpSPHScannerFixture'
    package.mkdir()
    (package/'__init__.py').write_text('')
    (package/'regular.py').write_text('present = 1\n')
    (package/'child.py').write_text('value = 2\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(module,'REPO',tmp_path)
    monkeypatch.setattr(module,'SRC',tmp_path/'src')
    try:
        yield module,tmp_path
    finally:
        for key in list(sys.modules):
            if key.startswith('warpSPHScannerFixture'):
                sys.modules.pop(key)


def test_missing_regular_module_symbol_reports_and_scan_continues(scanner):
    module,root=scanner
    (root/'use.py').write_text('from warpSPHScannerFixture.regular import absent\n'
                              'import warpSPHScannerFixture.missing\n')
    failures=module.static_pass(False)
    assert len(failures)==2
    assert failures[0].where=='use.py:1' and failures[0].detail=='symbol missing'
    assert "'absent'" in failures[0].what
    assert failures[1].where=='use.py:2' and 'missing' in failures[1].what


def test_package_submodule_and_regular_attribute_are_valid(scanner):
    module,root=scanner
    (root/'use.py').write_text('from warpSPHScannerFixture import child\n'
                              'from warpSPHScannerFixture.regular import present\n')
    assert module.static_pass(False)==[]


def test_broken_child_finder_reports_instead_of_aborting(scanner,monkeypatch):
    module,root=scanner
    (root/'use.py').write_text('from warpSPHScannerFixture import broken\n'
                              'from warpSPHScannerFixture.regular import absent\n')
    original=module.importlib.util.find_spec
    def lookup(name):
        if name=='warpSPHScannerFixture.broken':
            raise ValueError('broken finder metadata')
        return original(name)
    monkeypatch.setattr(module.importlib.util,'find_spec',lookup)
    failures=module.static_pass(False)
    assert len(failures)==2
    assert failures[0].detail=='ValueError: broken finder metadata'
    assert failures[1].detail=='symbol missing'


def test_parent_finder_exception_is_reported(scanner,monkeypatch):
    module,root=scanner
    (root/'use.py').write_text('import warpSPHScannerFixture.broken.child\n')
    original=module.importlib.util.find_spec
    def lookup(name):
        if name=='warpSPHScannerFixture.broken.child':raise RuntimeError('broken parent initialization')
        return original(name)
    monkeypatch.setattr(module.importlib.util,'find_spec',lookup)
    failures=module.static_pass(False)
    assert len(failures)==1 and failures[0].detail=='RuntimeError: broken parent initialization'


def test_installed_optional_package_missing_child_is_an_error(scanner,monkeypatch):
    module,root=scanner
    monkeypatch.setattr(module,'OPTIONAL_PACKAGES',{'warpSPHScannerFixture','warpSPHAbsentFixture'})
    (root/'use.py').write_text('import warpSPHScannerFixture.absent\nimport warpSPHAbsentFixture.absent\n')
    failures=module.static_pass(False)
    assert len(failures)==1 and 'warpSPHScannerFixture.absent' in failures[0].what
