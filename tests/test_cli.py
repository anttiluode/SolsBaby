import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class CLITests(unittest.TestCase):
    def run_cli(self, *args, env=None):
        return subprocess.run([sys.executable, '-m', 'solsbaby', *args],
                              text=True, capture_output=True, env=env, timeout=90)

    def test_help_explains_the_available_commands(self):
        p = self.run_cli('--help')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('benchmark', p.stdout)
        self.assertIn('demo', p.stdout)
        self.assertIn('gui', p.stdout)

    def test_demo_writes_valid_json_and_figure_to_directory_with_spaces(self):
        with tempfile.TemporaryDirectory(prefix='sols baby ') as tmp:
            p = self.run_cli('demo', '--out', tmp, '--seed', '3')
            self.assertEqual(p.returncode, 0, p.stderr)
            report = json.loads((Path(tmp)/'demo.json').read_text())
            self.assertEqual(report['branch'], 0)
            self.assertGreater(report['relative_response_change'], .01)
            self.assertGreater((Path(tmp)/'demo.png').stat().st_size, 10000)

    def test_small_benchmark_exports_machine_readable_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self.run_cli('benchmark', '--out', tmp, '--seeds', '3',
                             '--train-per-class', '2', '--test-per-class', '2')
            self.assertEqual(p.returncode, 0, p.stderr)
            result = json.loads((Path(tmp)/'results.json').read_text())
            self.assertEqual(len(result['trials']), 1)
            self.assertTrue((Path(tmp)/'episodes.csv').exists())
            self.assertTrue((Path(tmp)/'report.png').exists())
            self.assertIn('protocol_sha256', result)

    def test_invalid_numeric_arguments_and_missing_protocol_are_actionable(self):
        for args in (('benchmark', '--noise-std', '-1'),
                     ('benchmark', '--protocol', 'does-not-exist.json'),
                     ('demo', '--branch', '99')):
            with self.subTest(args=args):
                p = self.run_cli(*args)
                self.assertNotEqual(p.returncode, 0)
                self.assertNotIn('Traceback', p.stderr)

    @unittest.skipIf(sys.platform in ('win32', 'darwin'), 'Native GUI does not use DISPLAY')
    def test_missing_display_offers_headless_demo(self):
        env = dict(os.environ)
        env.pop('DISPLAY', None)
        env.pop('WAYLAND_DISPLAY', None)
        p = self.run_cli('gui', env=env)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('demo', p.stderr.lower())
        self.assertNotIn('Traceback', p.stderr)


if __name__ == '__main__':
    unittest.main()
