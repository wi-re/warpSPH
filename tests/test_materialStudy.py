"""A remotely stopped or failed probe must not launch another GPU child."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest


@pytest.mark.parametrize('code,status', [(130, 'stopped'), (1, 'failed')])
@pytest.mark.parametrize('script', ['study_materialEOS.py', 'study_materialCases.py'])
def test_study_stops_queue_on_child_stop_or_failure(tmp_path, monkeypatch, code, status, script):
    path = Path(__file__).resolve().parents[1]/'scripts'/script
    spec = importlib.util.spec_from_file_location('rm_study_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    launched = []
    class Child:
        def __init__(self, command, **kwargs):
            launched.append(command)
        def poll(self):
            return code
        def wait(self):
            return code
    monkeypatch.setattr(module.subprocess, 'Popen', Child)
    root = tmp_path/'campaign'
    monkeypatch.setattr(sys, 'argv', ['study_materialEOS.py', '--output', str(root)])
    assert module.main() == code
    assert len(launched) == 1
    manifest = json.loads((root/'campaign.json').read_text())
    assert manifest['status'] == status
    assert manifest['cases'][0]['status'] == status
    assert all(c['status'] == 'pending' for c in manifest['cases'][1:])
