#!/usr/bin/env python3
"""Join retained board/simulation/build evidence and save comparison figures."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if (run / 'report.json').exists():
        raise ValueError('Report already exists; preserve it')
    board = Path(__file__).resolve().parent
    repo = board.parents[2]
    read = lambda p: json.loads((run / p).read_text())
    hardware = read('hardware-isolated/metrics.json')
    rtl = read('rtl/metrics.json')
    transport = read('transport/metrics.json')
    fixtures = read('fixtures/fixtures.json')
    provenance = read('provenance.json')
    for result in [hardware, rtl, transport, fixtures]:
        assert result['status'] == 'passed'
    assert hardware['planned_jobs'] == len(hardware['jobs']) == 26
    assert all(j['passed'] and j['fault'] == 0 for j in hardware['jobs'])
    assert sha(run / 'build/vroom_inference.bit') == provenance['bitstream_sha256']
    for name, expected in provenance['source_sha256'].items():
        if Path(name).suffix in ['.v', '.sv', '.xdc']:
            assert sha(repo / name) == expected, f'Synthesized source changed: {name}'
    assert any(s.startswith('VROOM_PROGRAM_OK ') for s in (run / 'program.log').read_text().splitlines())
    assert any(s.startswith('VROOM_BUILD_OK ') for s in (run / 'vivado-build.log').read_text().splitlines())
    timing = (run / 'build/timing.rpt').read_text()
    match = re.search(r'\n\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+\d+\s+\d+\s+(-?\d+\.\d+)', timing)
    assert match
    setup, tns, hold = map(float, match.groups())
    assert setup >= 0 and tns == 0 and hold >= 0
    report = {'status': 'passed', 'completed_at': dt.datetime.now(dt.timezone.utc).isoformat(),
              'scope': hardware['scope'], 'hardware_programmed': True, 'storage': 'volatile JTAG',
              'checkpoint_sha256': hardware['checkpoint_sha256'],
              'vectors_sha256': hardware['golden_vectors_sha256'],
              'bitstream_sha256': provenance['bitstream_sha256'],
              'hardware_jobs': len(hardware['jobs']),
              'jobs_per_actor': {name: sum(j['actor'] == name for j in hardware['jobs']) for name in ['inner', 'outer']},
              'max_abs_error': hardware['max_abs_error'], 'atol': hardware['atol'],
              'clock_hz': provenance['clock_hz'], 'lanes': provenance['lanes'],
              'target': provenance['target'], 'cable_serial': provenance['cable_serial'],
              'idcode': provenance['idcode'], 'setup_slack_ns': setup, 'hold_slack_ns': hold,
              'lut_sites': 18146, 'flip_flops': 11992, 'dsp48e1': 24, 'bram36': 0,
              'rtl_regression': rtl['result'], 'transport': transport,
              'prior_hardware_attempt': {'status': read('hardware/metrics.json')['status'],
                  'path': 'hardware', 'reason': 'Competing synapse32 pyserial reader consumed a reply byte',
                  'process_pid': 1498889, 'action': 'User-approved SIGINT; original log preserved'},
              'control_rates_verified': False, 'model_or_precision_changed': False,
              'seed': 2805, 'golden_vector_seed': 90210,
              'versions': {'python': sys.version, 'matplotlib': matplotlib.__version__,
                  'vivado': '2024.2 build 5239630', 'iverilog': '13.0 stable', 'verilator': '5.052'},
              'platform': platform.platform(),
              'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
              'git_branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=repo, text=True).strip(),
              'submodules': subprocess.check_output(['git', 'submodule', 'status'], cwd=repo, text=True).strip()}
    snapshot = run / 'source/board'
    shutil.copytree(board, snapshot, ignore=shutil.ignore_patterns('__pycache__'))
    report['board_source_sha256'] = {str(p.relative_to(snapshot)): sha(p) for p in snapshot.rglob('*') if p.is_file()}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for name, color in [('inner', '#1672b4'), ('outer', '#d46d1f')]:
        jobs = [j for j in hardware['jobs'] if j['actor'] == name]
        axes[0].scatter([v for j in jobs for v in j['expected']],
                        [v for j in jobs for v in j['actual']],
                        label=f'{name}: 13 vectors / 26 outputs', color=color, s=27, alpha=.8)
    axes[0].plot([-1, 1], [-1, 1], color='#555555', ls='--', lw=1)
    axes[0].set(xlabel='Package golden raw output', ylabel='FPGA raw output', title='Both trained FP32 actors on Arty 35T')
    axes[0].legend(fontsize=8)
    axes[1].bar(range(1, 27), [j['max_abs_error'] for j in hardware['jobs']], color='#1672b4')
    axes[1].axhline(hardware['atol'], color='#ba3333', ls='--', label='Acceptance: 1e-5')
    axes[1].set_yscale('symlog', linthresh=1e-8)
    axes[1].set(xlabel='Hardware job (inner / outer alternating)', ylabel='Maximum absolute output difference', title=f'26/26 passed; maximum {hardware["max_abs_error"]:.2g}')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(run / 'comparison.png', dpi=160)
    fig.savefig(run / 'comparison.svg')
    plt.close(fig)
    summary = f'''# Arty trained RL inference comparison

**Passed: 26/26 complete physical FPGA inference jobs**, 13 each for the trained
inner and outer actors. Maximum raw-output absolute error: {hardware['max_abs_error']:.10g};
predeclared package tolerance: 1e-5 (relative tolerance zero).

The unchanged FP32 networks ran through the existing 12-lane stream, two ReLU
hidden layers and tanh outputs. The supplied checkpoint SHA256 is
`{hardware['checkpoint_sha256']}`. No weights were retrained, resized or quantized.

The actual Arty device was xc7a35t, IDCODE 0362D093, cable 210319A43D3E.
Vivado 2024.2 routed xc7a35ticsg324-1L at 100 MHz: setup slack {setup:.3f} ns,
hold slack {hold:.3f} ns, no failing internal timing endpoints. The design uses
18,146/20,800 LUT sites, 11,992 flip-flops, 24 DSP48E1 and zero BRAM36.
JTAG programming read back VRM2 (56524d32), DONE=1 and configuration CRC error=0.
SPI flash was preserved. The inference design remains loaded until power cycle
or reprogramming.

The package software check passed all 518 golden vectors for each actor. The
trained-weight RTL regression checked 2,056 intermediate layer values exactly,
four complete jobs, source/output stalls and 16 fault/reset protocol cases.
UART simulation checked all 256 byte values both ways at the production divider.
Transport isolation checked 231,873 words and generated metadata against real
request packets, verified reply CRCs and explicitly rejected a corrupt CRC.

The first hardware attempt remains failed under `hardware/`: two valid results,
then a 31-byte reply. A concurrent synapse32 test reader (PID 1498889) held the
same ttyUSB1 open. Its user-approved SIGINT released the port; its log was kept.
Both flock and TIOCEXCL are now required. The new run under `hardware-isolated/`
passed all 26 jobs without inference reset/retry or reconstructed reply bytes.

This verifies prepared-observation-to-raw-output inference logic. Full weights
are sent over 1 Mbaud UART for every job, taking about three seconds per request.
The reported hardware cycle counter includes UART starvation. DDR/cache,
periodic 10 kHz/1 kHz deadlines, controller history/action adapters and motor
operation were not exercised.

Raw transmit packets: `fixtures/board-job*.bin`. Raw replies and comparisons:
`hardware-isolated/reply-*.bin`, `trace.jsonl`, `metrics.json`. Source snapshots,
versions and SHA256 values: `rtl/`, `source/`, `provenance.json`, `report.json`,
`artifact_hashes.json`. Physical reports and bitstream: `build/`. Plot:
`comparison.png` / `comparison.svg`.

Reproduction from the isolated `codex/arty-inference-board` worktree is documented
in `hw/arty/board/README.md`. The original `main` checkout is unchanged.
'''
    (run / 'summary.md').write_text(summary)
    (run / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    state = read('status.json')
    state.update(status='passed', hardware_programmed=True, completed_at=report['completed_at'],
                 hardware_jobs=26, max_abs_error=hardware['max_abs_error'])
    (run / 'status.json').write_text(json.dumps(state, indent=2) + '\n')
    (run / 'artifact_hashes.json').write_text(json.dumps({str(p.relative_to(run)): sha(p)
        for p in run.rglob('*') if p.is_file() and p.name != 'artifact_hashes.json'}, indent=2) + '\n')
    print(json.dumps({'status': 'passed', 'summary': str(run / 'summary.md'),
                      'hardware_jobs': 26, 'max_abs_error': hardware['max_abs_error']}, indent=2))


if __name__ == '__main__':
    main()
