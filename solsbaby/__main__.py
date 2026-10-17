"""Command-line entry point; GUI dependencies are imported only on demand."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import platform
import subprocess
import sys

from . import __version__
from .experiment import Protocol, run_benchmark, walkthrough


def main(argv=None):
    parser = argparse.ArgumentParser(description='SolsBaby: experience changes a dendritic response operator.')
    commands = parser.add_subparsers(dest='command')
    gui = commands.add_parser('gui', help='Open the interactive teach / probe / erase demo')
    gui.add_argument('--seed', type=int, default=3)
    demo = commands.add_parser('demo', help='Save a visual walkthrough without a display')
    demo.add_argument('--out', type=Path, default=Path('runs/demo'))
    demo.add_argument('--seed', type=int, default=3)
    demo.add_argument('--branch', type=int, default=0)
    demo.add_argument('--active', action='store_true', help='Enable the illustrative voltage-dependent boost')
    bench = commands.add_parser('benchmark', help='Run held-out memory measurement and matched controls')
    bench.add_argument('--out', type=Path, default=Path('runs/benchmark'))
    bench.add_argument('--protocol', type=Path)
    bench.add_argument('--seeds', nargs='+', type=int)
    bench.add_argument('--train-per-class', type=int)
    bench.add_argument('--test-per-class', type=int)
    bench.add_argument('--noise-std', type=float)
    bench.add_argument('--fail-on-gate', action='store_true', help='Exit with code 2 when the research gate fails')
    args = parser.parse_args(argv)
    try:
        if args.command == 'gui':
            try:
                from .gui import launch
            except ImportError as exc:
                raise RuntimeError('GUI needs Tkinter and Matplotlib. Install Tk for your Python, or use: python -m solsbaby demo') from exc
            launch(args.seed)
        elif args.command == 'demo':
            from .report import save_demo
            result = save_demo(args.out, walkthrough(args.seed, args.branch, args.active))
            print(json.dumps(result, indent=2))
            print(f'Saved {args.out / "demo.png"}')
        elif args.command == 'benchmark':
            from .report import save_benchmark
            p = Protocol(**json.loads(args.protocol.read_text(encoding='utf-8'))) if args.protocol else Protocol()
            overrides = {name: getattr(args, name) for name in ('seeds', 'train_per_class', 'test_per_class', 'noise_std')
                         if getattr(args, name) is not None}
            if overrides:
                p = replace(p, **overrides)
            p.validate()
            print(f'Protocol SHA256: {p.fingerprint}', flush=True)
            def progress(trial):
                print(f"seed {trial['seed']}: learned soma {trial['accuracy']['learned_soma']:.1%}; "
                      f"frozen {trial['accuracy']['frozen_writer']:.1%}; erased {trial['accuracy']['erased_memory']:.1%}", flush=True)
            result = run_benchmark(p, progress=progress)
            result['registered_v1'] = p.fingerprint == Protocol().fingerprint
            result['software'] = {'solsbaby': __version__, 'python': platform.python_version()}
            root = Path(__file__).resolve().parent.parent
            if (root/'.git').exists():
                result['source_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
                result['source_dirty'] = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip())
            save_benchmark(args.out, result)
            print(f"Gate: {result['gate']['status']} · {result['seconds']:.2f}s · saved {args.out / 'report.png'}")
            if args.fail_on_gate and result['gate']['status'] != 'PASS':
                return 2
        else:
            parser.print_help()
    except (ValueError, TypeError, OSError, RuntimeError, ImportError) as exc:
        print(f'SolsBaby: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
