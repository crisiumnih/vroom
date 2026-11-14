#!/usr/bin/env python3
"""Prove a specialized FP32 combinational block equal to pinned HardFloat."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unit',choices=['recode','round-mask'],default='recode')
    args=parser.parse_args()
    module='vroom_fp32_recode' if args.unit=='recode' else 'vroom_fp32_round_mask'
    input_width,output_width=(32,33) if args.unit=='recode' else (9,25)
    reference='fNToRecFN #(8,24)' if args.unit=='recode' else 'lowMaskLoHi #(9,105,130)'
    src = Path(__file__).resolve().parents[1]
    repo = src.parents[2]
    root = repo / 'results/arty_fp32'
    root.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=time.strftime(args.unit+'-proof-%Y%m%dT%H%M%S-'), dir=root))
    print(out, flush=True)
    report = {'status': 'running', 'scope': __doc__, 'python': platform.python_version(),
              'platform': platform.platform(), 'commands': [], 'seed': None,
              'configuration': f'Combinational SAT equivalence, unconstrained {input_width}-bit input',
              'unit':args.unit}
    def run(name, command):
        started = time.monotonic()
        result = subprocess.run(command, cwd=out, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=120)
        (out / (name + '.log')).write_text(result.stdout)
        report['commands'].append({'name': name, 'argv': command, 'returncode': result.returncode,
                                   'seconds': time.monotonic() - started})
        if result.returncode:
            raise RuntimeError(f'{name} failed; see {out}')
        return result.stdout
    try:
        shutil.copy2(__file__, out / 'prove_recode.py')
        shutil.copy2(src / ('rtl/'+module+'.v'), out / (module+'.v'))
        shutil.copytree(repo / 'third_party/hardfloat', out / 'hardfloat')
        for name, command in {'yosys': ['yosys', '-V'], 'git-head': ['git', '-C', str(repo), 'rev-parse', 'HEAD'],
                              'git-branch': ['git', '-C', str(repo), 'branch', '--show-current'],
                              'git-status': ['git', '-C', str(repo), 'status', '--short'],
                              'git-submodules': ['git', '-C', str(repo), 'submodule', 'status']}.items():
            report[name] = run(name, command)
        (out / 'miter.v').write_text(f'module miter(input [{input_width-1}:0] in, output mismatch);\n'
                                    f'wire [{output_width-1}:0] actual,expected;\n'
                                    f'{module} dut(in,actual);\n'
                                    f'{reference} golden(in,expected);\n'
                                    'assign mismatch=actual!=expected;\nendmodule\n')
        (out / 'proof.ys').write_text('read_verilog -Ihardfloat/source hardfloat/source/HardFloat_primitives.v '
                                     f'hardfloat/source/fNToRecFN.v {module}.v miter.v\n'
                                     'prep -top miter -flatten\n'
                                     'sat -verify -prove mismatch 0 -show-inputs -show-outputs\n')
        report['source_hashes'] = {str(p.relative_to(out)): digest(p) for p in out.rglob('*')
                                    if p.is_file() and p.suffix != '.log'}
        log = run('proof', ['yosys', '-s', 'proof.ys'])
        assert 'SAT proof finished - no model found: SUCCESS!' in log
        report['status'] = 'passed'
    except BaseException as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        (out / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
        (out / 'summary.txt').write_text(f"Status: {report['status']}\n{report['scope']}\n")
        (out / 'artifact_hashes.json').write_text(json.dumps({str(p.relative_to(out)): digest(p)
            for p in out.rglob('*') if p.is_file() and p.name != 'artifact_hashes.json'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
