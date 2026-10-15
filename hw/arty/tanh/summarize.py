#!/usr/bin/env python3
"""Join tanh simulation, routed diagnostic and the retained dense compute budget.
The combined budgets are conditional: no scheduler, DDR or model-parity claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['verification','probe','compute']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();folders={k:v.resolve() for k,v in vars(args).items()}
    root=Path(__file__).resolve().parents[3]/'results/arty_tanh';root.mkdir(parents=True,exist_ok=True)
    out=Path(tempfile.mkdtemp(prefix=time.strftime('report-%Y%m%dT%H%M%S-'),dir=root))
    report={'status':'running','scope':'Conditional scheduling arithmetic, not an integrated controller',
            'sources':{k:{'directory':str(v),'metrics_sha256':digest(v/'metrics.json')} for k,v in folders.items()},
            'cases':[],'full_deadline_proven':False}
    shutil.copy2(__file__,out/'summarize.py')
    try:
        data={k:json.loads((v/'metrics.json').read_text()) for k,v in folders.items()}
        assert data['verification']['status']=='passed'
        assert digest(folders['verification']/'source/rtl/vroom_fp32_tanh.v')==digest(folders['probe']/'source/vroom_fp32_tanh.v')
        report['tested_and_routed_rtl_identical']=True
        case=next(c for c in data['probe']['cases'] if c['name']=='fptanh');cells=case['dut_mapped_cells']
        report['physical_probe']={'status':case['status'],
            'diagnostic_fmax_mhz':min(v['achieved'] for v in case['pnr']['fmax'].values()),
            'dut_logic_lut_cells':sum(cells.get('LUT'+str(i),0) for i in range(1,7)),
            'dut_ff_cells':sum(v for k,v in cells.items() if k.startswith('FD')),
            'dsp48e1':cells.get('DSP48E1',0),'bram36':cells.get('RAMB36E1',0),
            'database_capacity_mismatch':case['database_capacity_mismatch']}
        report['verification']=data['verification']['rtl'];report['accuracy']=data['verification']['rtl_ulp_histogram']
        tanh=data['verification']['budget_at_100mhz'];report['tanh_budget']=tanh
        assert data['compute']['config']['mhz']==100
        tail=tanh['two_outputs_cycles']
        for c in data['compute']['cases']:
            inner=c['actors']['inner']['cycles'];outer=c['actors']['outer']['cycles']
            report['cases'].append({'lanes':c['lanes'],'feedback_cycles':c['feedback_cycles'],
                'inner_dense_us':inner/100,'inner_dense_plus_tanh_us':(inner+tail)/100,
                'outer_dense_us':outer/100,'outer_dense_plus_tanh_us':(outer+tail)/100,
                'dense_occupancy_percent':c['compute_load_percent'],
                'serialized_dense_and_tanh_occupancy_percent':(10*(inner+tail)+outer+tail)/1000,
                'separate_tanh_unit_occupancy_percent':tanh['separate_unit_utilization_percent']})
        report['assumptions']=['100 MHz','two output activations per actor job',
            '10 inner and one outer job per ms','zero job-boundary overhead',
            'no activation queue wait or output stall in tail estimate',
            'serialized occupancy applies only if dense waits for tanh before starting another job',
            'separate-resource occupancy is necessary but not sufficient for schedulability',
            'DDR, ReLU passes, scaling, adapter, arbitration and commits excluded']
        report['status']='completed'
    except BaseException as exc:report.update(status='failed',error=str(exc));raise
    finally:
        (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
        lines=['# CORDIC tanh: measured unit and conditional controller cost','',f'Status: {report["status"]}','',report['scope'],'']
        if 'physical_probe' in report:lines += ['```json',json.dumps(report['physical_probe'],indent=2),'```','']
        lines += ['| Lanes | Dense inner us | With tanh tail us | Fully serialized 1 ms occupancy |', '| --- | ---: | ---: | ---: |']
        for c in report['cases']:lines.append(f'| {c["lanes"]} | {c["inner_dense_us"]:.2f} | {c["inner_dense_plus_tanh_us"]:.2f} | {c["serialized_dense_and_tanh_occupancy_percent"]:.3f}% |')
        lines += ['', 'A separate tanh unit can overlap dense work; its occupancy must not be added to dense occupancy unless the scheduler serializes the two. No complete scheduling/DDR proof is provided. The open chip database maps the 35T onto larger 50T fabric; Fmax is diagnostic only.','']
        (out/'summary.md').write_text('\n'.join(lines))
        (out/'artifact_hashes.json').write_text(json.dumps({f.name:digest(f) for f in out.iterdir() if f.is_file() and f.name!='artifact_hashes.json'},indent=2)+'\n');print(out)


if __name__=='__main__':main()
