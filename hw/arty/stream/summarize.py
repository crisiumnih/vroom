#!/usr/bin/env python3
"""Join complete-inference simulation and mapping/route evidence without editing it.
Cycle-to-time conversion assumes 100 MHz; failed routes never become timing passes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verification', type=Path, required=True)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--figures', action='store_true', help='Create timing/budget figures; requires matplotlib')
    args = parser.parse_args()
    folders = {key: getattr(args,key).resolve() for key in ('verification','probe')}
    root = Path(__file__).resolve().parents[3] / 'results/arty_stream'
    root.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=time.strftime('report-%Y%m%dT%H%M%S-'), dir=root))
    shutil.copy2(__file__, out / 'summarize.py')
    report = {'status': 'running', 'full_deadline_proven': False, 'cases': [],
              'scope': 'Synthetic serial inference; cache/DDR and periodic scheduling excluded',
              'sources': {k: {'directory': str(v), 'metrics_sha256': digest(v / 'metrics.json')}
                          for k, v in folders.items()}}
    try:
        sim = json.loads((folders['verification'] / 'metrics.json').read_text())
        probe = json.loads((folders['probe'] / 'metrics.json').read_text())
        assert sim['status'] == 'passed'
        assert probe['status'] in ('completed', 'completed_with_failures')
        pairs = []
        for unit in ('stream', 'fp32', 'tanh'):
            for rtl in (folders['verification'] / 'source' / unit / 'rtl').glob('*.v'):
                pairs.append((rtl, folders['probe'] / 'source' / rtl.name))
        for vendor in (folders['verification'] / 'hardfloat').rglob('*'):
            if vendor.is_file():
                pairs.append((vendor, folders['probe'] / 'source/hardfloat' /
                              vendor.relative_to(folders['verification'] / 'hardfloat')))
        assert pairs and all(digest(a) == digest(b) for a, b in pairs)
        report['tested_and_probed_sources_identical'] = True
        report['matched_source_files'] = len(pairs)
        report['catalog_capacity'] = probe['catalog_capacity']
        report['database_note'] = 'Installed nominal 35T chipdb uses larger 50T fabric; no 35T sign-off'
        report['simulation_widths_without_physical_probe'] = []
        for tested in sim['checks']:
            case = next((c for c in probe['cases'] if c['name'] == f"stream{tested['lanes']}"), None)
            if case is None:
                report['simulation_widths_without_physical_probe'].append(tested['lanes'])
                continue
            cells = case['dut_mapped_cells']
            jobs = tested['jobs_measured']
            inner = next(j for j in jobs if j['actor'] == 0 and j['mode'] == 0)['latency']
            outer = next(j for j in jobs if j['actor'] == 1 and j['mode'] == 0)['latency']
            route_log = folders['probe'] / (case['name'] + '-nextpnr.log')
            log = route_log.read_text() if route_log.exists() else ''
            pre_route = log.split('Running router2...')[0]
            estimates = re.findall(r"Max frequency for clock '[^']+': ([\d.]+) MHz", pre_route)
            constant_failures = re.findall(r'ERROR: (\d+) constant sink\(s\) could not be routed', log)
            routed_clocks = (case.get('pnr', {}).get('fmax')
                             if case['status'] in ('timing_pass', 'timing_failed_or_unreported') else None)
            routed_mhz = min((c['achieved'] for c in (routed_clocks or {}).values()), default=None)
            report['cases'].append({
                'lanes': tested['lanes'], 'simulation': tested,
                'serial_10_inner_plus_1_outer_cycles': 10 * inner + outer,
                'serial_occupancy_percent_at_assumed_100mhz': (10 * inner + outer) / 1000,
                'handover_and_output_waits_in_occupancy': False,
                'dut_logic_lut_cells': sum(cells.get('LUT' + str(i), 0) for i in range(1, 7)),
                'dut_ff_cells': sum(v for k, v in cells.items() if k.startswith('FD')),
                'dut_yosys_estimated_lcs': case.get('dut_yosys_estimated_lcs'),
                'dut_dsp48e1': cells.get('DSP48E1', 0),
                'dut_bram36': cells.get('RAMB36E1', 0),
                'dut_memory_primitives': {k: v for k, v in cells.items() if k.startswith(('RAM', 'SRL'))},
                'probe_status': case['status'],
                'pre_route_estimate_mhz_not_accepted_timing': [float(v) for v in estimates],
                'constant_sinks_unrouted': [int(v) for v in constant_failures],
                'routed_clock_report': routed_clocks,
                'target_clock_met': case['status'] == 'timing_pass',
                'inner_us_at_routed_diagnostic_fmax': inner / routed_mhz if routed_mhz else None,
                'accepted_routed_fmax': case.get('pnr', {}).get('fmax') if case['status'] == 'timing_pass' else None,
            })
        if args.figures:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            report['matplotlib_version']=matplotlib.__version__
            fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
            rows=report['cases']; positions=list(range(len(rows)))
            clocks=[min((v['achieved'] for v in (c['routed_clock_report'] or {}).values()),default=0) for c in rows]
            bars=axes[0].bar(positions,clocks,color=['#207a64' if c['target_clock_met'] else '#ad543e' for c in rows])
            axes[0].bar_label(bars,labels=[f'{v:.2f}' if v else 'unavailable' for v in clocks],padding=4)
            axes[0].axhline(probe['target_mhz'],color='#323e4a',ls='--',label=f"Target {probe['target_mhz']:g} MHz")
            axes[0].set(xticks=positions,xticklabels=[f"{c['lanes']} lanes" for c in rows],ylabel='Routed diagnostic clock (MHz)',title='Clock constraint',ylim=(0,max([probe['target_mhz']]+clocks)*1.2))
            for mode,label,offset,color in [(0,'Ready source',-.18,'#207a64'),(1,'Injected source gaps',.18,'#ad543e')]:
                values=[next(j['latency']/100 for j in c['simulation']['jobs_measured'] if j['actor']==0 and j['mode']==mode) for c in rows]
                bars=axes[1].bar([x+offset for x in positions],values,width=.34,label=label,color=color)
                axes[1].bar_label(bars,labels=[f'{v:.2f}' for v in values],padding=-14,fontsize=9,color='white')
            axes[1].axhline(100,color='#323e4a',ls='--',label='Inner deadline: 100 us')
            axes[1].set(xticks=positions,xticklabels=[f"{c['lanes']} lanes" for c in rows],ylabel='Inner latency (us at assumed 100 MHz)',title='Job admission to complete output',ylim=(0,150))
            for ax in axes:
                ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=8,loc='lower left')
            fig.suptitle('Complete FP32 actor stream — open-flow diagnostic, no 35T sign-off',fontsize=12)
            fig.savefig(out/'timing.png',dpi=160);fig.savefig(out/'timing.svg');plt.close(fig)
        report['status'] = 'completed'
    except BaseException as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        (out / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
        lines = ['# Complete serial inference: verification and physical probe', '',
                 f'Status: {report["status"]}', '', report['scope'], '',
                 '| Lanes | Inner clocks | Serial 1 ms occupancy at assumed 100 MHz | Logic LUT cells | Routed MHz | Route |',
                 '| --- | ---: | ---: | ---: | ---: | --- |']
        for c in report['cases']:
            inner = next(j['latency'] for j in c['simulation']['jobs_measured'] if not j['actor'] and not j['mode'])
            mhz=min((v['achieved'] for v in (c['routed_clock_report'] or {}).values()),default=None)
            lines.append(f'| {c["lanes"]} | {inner} | {c["serial_occupancy_percent_at_assumed_100mhz"]:.3f}% | {c["dut_logic_lut_cells"]} | {mhz if mhz is not None else "unavailable"} | {c["probe_status"]} |')
        lines += ['', 'Occupancy extrapolates the two ready-source synthetic jobs and excludes handover/output waits; it is not a periodic scheduling proof. Logic cells and Yosys LC estimates are not physical site counts. Pre-route timing from failed runs is diagnostic only. The installed open chipdb uses larger 50T fabric. No complete 35T fit or 100 MHz sign-off is claimed.', '']
        for name, path in folders.items():
            lines.append(f'{name}: `{path}`')
        if (out/'timing.png').exists():lines+=['','![Clock and latency evidence](timing.png)']
        (out / 'summary.md').write_text('\n'.join(lines) + '\n')
        (out / 'artifact_hashes.json').write_text(json.dumps({p.name: digest(p) for p in out.iterdir()
                                                             if p.is_file() and p.name != 'artifact_hashes.json'}, indent=2) + '\n')
        print(out)


if __name__ == '__main__':
    main()
