"""Protocol/analysis checks; stdlib only, no runtime or GPU imports."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import analyze
import run


class HarnessTests(unittest.TestCase):
    def request(self, *args):
        return list(run.requests(run.parser().parse_args(args)))

    def test_plan_never_imports_runtime(self):
        code = "import run, sys; run.main(['--plan', '--suite', 'memory']); assert 'torch' not in sys.modules; assert 'warpSPHCore' not in sys.modules"
        p = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).parent, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(len(json.loads(p.stdout)), 3)

    def test_layout_execution_and_topology_are_real_sweep_axes(self):
        requests = self.request('--suite', 'steps', '--schemes', 'CRKSPH', '--execution', 'eager', 'auto', '--lanes', '1', '8')
        self.assertEqual(len(requests), 16)
        self.assertEqual(len({analyze.key(r) for r in requests}), 16)

    def test_irrelevant_memory_axes_dont_duplicate_cases(self):
        requests = self.request('--suite', 'memory', '--nx', '16', '32', '--particle-order', 'input', 'morton', '--execution', 'auto', 'graph')
        self.assertEqual(len(requests), 3)

    def test_match_devices_but_not_numerics_or_layout(self):
        r = self.request('--suite', 'operators', '--operators', 'gradient')[0]
        self.assertEqual(analyze.key(r), analyze.key(dict(r, device='cuda:0', repeats=99, profile=True)))
        for change in (dict(precision='float64'), dict(warmup=99), dict(neighbors=99),
                       dict(particle_order='morton'), dict(threads=8), dict(offset_dtype='int32')):
            self.assertNotEqual(analyze.key(r), analyze.key(dict(r, **change)))
        self.assertNotEqual(analyze.key(r), analyze.key(dict(r, backend='native')))
        self.assertEqual(analyze.key(r, allow_backend_change=True), analyze.key(dict(r, backend='native'), allow_backend_change=True))

    def test_capacity_screen_and_nonfinite_budget(self):
        r = self.request('--suite', 'steps', '--schemes', 'CRKSPH', '--nx', '128')[0]
        self.assertGreater(run.estimated_bytes(r), 24*1024**3)
        with self.assertRaises(ValueError):
            self.request('--budget-gib', 'nan')

    def test_analysis_does_not_pair_failures_or_different_orders(self):
        r = self.request('--suite', 'steps', '--schemes', 'CRKSPH')[0]
        def record(request, ms, status='ok'):
            return dict(request=request, status=status, timing=dict(wall_ms=dict(median=ms, min=ms, max=ms),
                        stream_ms=None, samples=[dict(wall_ms=ms)]))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label, data in [('a', [record(r, 10), record(dict(r, particle_order='morton'), 5)]),
                                ('b', [record(dict(r, device='cuda:0'), 2), record(dict(r, particle_order='morton'), 1, 'failed')])]:
                (root/f'{label}.json').write_text(json.dumps(dict(schema_version=1, results=data)))
            with contextlib.redirect_stdout(io.StringIO()):
                analyze.main([str(root/'a.json'), str(root/'b.json'), '--output', str(root/'report')])
            paired = json.loads((root/'report/comparison.json').read_text())
            self.assertEqual(len(paired), 1)
            self.assertEqual(paired[0]['wall_speedup'], 5)
            tuning = json.loads((root/'report/compute_flags.json').read_text())
            self.assertEqual(len(tuning), 1)
            self.assertEqual(tuning[0]['wall_speedup'], 2)

    def test_worker_exit_code_overrides_partial_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            helper = root/'scripts/benchmarks/bench_single_device.py'
            helper.parent.mkdir(parents=True)
            helper.write_text("import sys,json,pathlib\na=sys.argv\nr=json.loads(pathlib.Path(a[a.index('--request')+1]).read_text())\npathlib.Path(a[a.index('--output')+1]).write_text(json.dumps(dict(schema_version=1,request=r,status='ok')))\nsys.exit(7)\n")
            with contextlib.redirect_stdout(io.StringIO()):
                code = run.main(['--core-root', str(root), '--output', str(root/'out'), '--suite', 'operators',
                                 '--operators', 'density', '--particle-order', 'input'])
            self.assertEqual(code, 1)
            result = json.loads((root/'out/results.json').read_text())['results'][0]
            self.assertEqual(result['status'], 'failed')
            self.assertIn('7', result['error'])

    def test_capacity_skip_does_not_start_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            helper = root/'scripts/benchmarks/bench_single_device.py'
            helper.parent.mkdir(parents=True)
            helper.touch()
            with patch('run.subprocess.Popen') as popen, contextlib.redirect_stdout(io.StringIO()):
                code = run.main(['--core-root', str(root), '--output', str(root/'out'), '--suite', 'steps',
                                 '--schemes', 'CRKSPH', '--particle-order', 'input', '--neighborhood', 'csr', '--nx', '128'])
            self.assertEqual(code, 0)
            popen.assert_not_called()
            self.assertEqual(json.loads((root/'out/results.json').read_text())['results'][0]['status'], 'skipped_capacity')

    def test_timeout_retains_failure_and_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            helper = root/'scripts/benchmarks/bench_single_device.py'
            helper.parent.mkdir(parents=True)
            helper.write_text('import time\nprint("started", flush=True)\ntime.sleep(30)\n')
            with contextlib.redirect_stdout(io.StringIO()):
                code = run.main(['--core-root', str(root), '--output', str(root/'out'), '--suite', 'operators',
                                 '--operators', 'density', '--particle-order', 'input', '--timeout', '1'])
            self.assertEqual(code, 1)
            result = json.loads((root/'out/results.json').read_text())['results'][0]
            self.assertEqual(result['status'], 'failed')
            self.assertIn('timeout=1', result['error'])
            self.assertIn('started', next((root/'out/logs').glob('*.log')).read_text())


if __name__ == '__main__':
    unittest.main()
