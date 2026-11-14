#!/usr/bin/env python3
"""Run existing full-layer/protocol RTL regression with trained-weight fixtures."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixtures', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--lanes', type=int, default=12, choices=[12, 16])
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    arty = Path(__file__).resolve().parents[1]
    repo = arty.parents[1]
    report = {'status': 'running', 'commands': [], 'source_hashes': {}, 'lanes': a.lanes}
    def run(name, argv):
        start = time.monotonic()
        proc = subprocess.run([str(v) for v in argv], cwd=a.out, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, timeout=600)
        (a.out / (name + '.log')).write_text(proc.stdout)
        report['commands'].append({'argv': [str(v) for v in argv], 'returncode': proc.returncode,
                                   'seconds': time.monotonic() - start})
        if proc.returncode:
            raise RuntimeError(f'{name} failed: {a.out / (name + ".log")}')
        return proc.stdout
    try:
        for directory in ['stream', 'fp32', 'tanh']:
            shutil.copytree(arty / directory, a.out / 'source' / directory,
                            ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(repo / 'third_party/hardfloat', a.out / 'hardfloat')
        for f in a.fixtures.glob('*.hex'):
            shutil.copy2(f, a.out / f.name)
        source = a.out / 'source'
        vendor = a.out / 'hardfloat'
        rtl = [f for folder in ['stream', 'fp32', 'tanh'] for f in sorted((source / folder / 'rtl').glob('*.v'))]
        lib = [*sorted((vendor / 'source').glob('*.v')), vendor / 'source/RISCV/HardFloat_specialize.v']
        inc = ['-I' + str(vendor / 'source'), '-I' + str(vendor / 'source/RISCV')]
        for root in [source, vendor]:
            for f in root.rglob('*'):
                if f.is_file():
                    report['source_hashes'][str(f.relative_to(a.out))] = hashlib.sha256(f.read_bytes()).hexdigest()
        run('lint', ['verilator', '--lint-only', '--Wall', '--top-module', 'vroom_actor_stream',
                     f'-GLANES={a.lanes}', source / 'fp32/sim/vendor_lint.vlt', *inc, *lib, *rtl])
        run('compile', ['iverilog', '-g2005', '-Wall', '-s', 'tb_actor_stream',
                        f'-Ptb_actor_stream.LANES={a.lanes}', '-o', a.out / 'test.vvp',
                        *inc, *lib, *rtl, source / 'stream/sim/tb_actor_stream.v'])
        log = run('simulate', ['vvp', a.out / 'test.vvp', '+seed=2805'])
        result = [json.loads(s[7:]) for s in log.splitlines() if s.startswith('RESULT ')]
        assert len(result) == 1 and result[0]['status'] == 'passed'
        report.update(status='passed', result=result[0])
    except BaseException as error:
        report.update(status='failed', error=str(error))
    finally:
        (a.out / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: v for k, v in report.items() if k not in ['source_hashes', 'commands']}, indent=2), flush=True)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
