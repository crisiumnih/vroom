#!/usr/bin/env python3
"""Join retained numerical, physical-probe and conditional-budget evidence.
No new simulation or device-fit claim; preserve references/hashes to raw inputs.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import time


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--simulation',type=Path,required=True)
    p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--budget',type=Path,required=True)
    args=p.parse_args()
    folders={k:v.resolve() for k,v in vars(args).items()}
    inputs={k:json.loads((v/'metrics.json').read_text()) for k,v in folders.items()}
    root=Path(__file__).resolve().parents[3]/'results/arty_fp32'
    out=Path(tempfile.mkdtemp(prefix=time.strftime('report-%Y%m%dT%H%M%S-'),dir=root))
    report={'status':'running','scope':'One-lane diagnostic, no full-controller fit/deadline claim',
            'inputs':{k:{'directory':str(v),'metrics_sha256':digest(v/'metrics.json')} for k,v in folders.items()},
            'cases':[],'simulation_checks':inputs['simulation']['checks'],
            'budget_config':inputs['budget']['config'],'conditional_budget':inputs['budget']['cases']}
    shutil.copy2(__file__,out/'summarize.py')
    try:
        assert inputs['simulation']['status']=='passed'
        # The numerical and routed implementations must be byte-identical.
        for f in (folders['simulation']/'source/rtl').glob('*.v'):
            assert digest(f)==digest(folders['probe']/'source'/f.name),f.name
        report['numerical_and_routed_rtl_identical']=True
        for c in inputs['probe']['cases']:
            t=c['dut_mapped_cells']
            estimated=re.findall(r'Estimated number of LCs:\s+(\d+)',(folders['probe']/(c['name']+'-dut-synth.log')).read_text())
            report['cases'].append({'name':c['name'],'status':c['status'],
                'fmax_mhz':min(v['achieved'] for v in c['pnr']['fmax'].values()),
                'logic_lut_cells':sum(t.get('LUT'+str(i),0) for i in range(1,7)),
                'flipflops':sum(v for k,v in t.items() if k.startswith('FD')),
                'dsp48':t.get('DSP48E1',0),'bram36':t.get('RAMB36E1',0),
                'ram32m':t.get('RAM32M',0),'ram64m':t.get('RAM64M',0),'srl16e':t.get('SRL16E',0),
                'yosys_estimated_logic_cells':int(estimated[-1]),
                'database_capacity_mismatch':c['database_capacity_mismatch']})
        # Model M includes multiplier transfer plus the product FIFO write/read
        # boundary. Its retirement endpoint is row availability, one cycle later
        # than the RTL last-write trace; F is the measured repeated-row interval.
        cfg=inputs['budget']['config']
        assert cfg['mul_latency']==7 and cfg['feedback']==[11]
        for c in report['simulation_checks']:
            if c['name'].startswith('lane'):
                g=c['groups'];predicted=255*max(g,11)+(g-1)+7+11
                assert predicted==c['cycles_first_issue_to_last_write']+1
        report['budget_crosscheck']='All five 256-column lane cases match model minus one final visibility edge'
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        report['matplotlib_version']=matplotlib.__version__
        fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
        values=[c['fmax_mhz'] for c in report['cases']]
        axes[0].bar(['Multiply','Add','Whole lane'],values,color=['#517fa6','#517fa6','#2d8073'])
        axes[0].axhline(100,color='black',ls='--',label='100 MHz target')
        for i,v in enumerate(values):axes[0].text(i,v+1,f'{v:.2f}',ha='center')
        axes[0].set(ylabel='Routed diagnostic Fmax (MHz)',ylim=(0,140),title='Single lane; scan fixture; seed 2805')
        axes[0].legend(fontsize=8)
        for j,groups in enumerate([1,16,22]):
            with (folders['simulation']/f'lane{groups}/lane-events.csv').open() as f:
                rows=list(csv.DictReader(f))
            origin=int(next(r['cycle'] for r in rows if r['event']=='issue'))
            cycles=[int(r['cycle'])-origin for r in rows if r['event']=='retire' and int(r['cycle'])-origin<145]
            axes[1].scatter(cycles,[j]*len(cycles),marker='|',s=90,label=f'{groups} rows')
        axes[1].set(yticks=[0,1,2],yticklabels=['1 row','16 rows','22 rows'],xlabel='Clocks since first request',title='Observed accumulator writes',ylim=(-.5,2.5))
        for ax in axes:ax.spines[['top','right']].set_visible(False)
        fig.suptitle('FP32 lane: numerical tests pass; backend uses larger fabric than the actual 35T',fontsize=11)
        fig.savefig(out/'lane-evidence.png',dpi=160);fig.savefig(out/'lane-evidence.svg');plt.close(fig)
        report['status']='completed'
    except BaseException as exc:
        report.update(status='failed',error=str(exc));raise
    finally:
        (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
        lines=['# FP32 lane evidence','',report['scope'],'',f'Status: {report["status"]}','',
               '| Unit | Diagnostic MHz | Logic LUT cells | FF | DSP |', '| --- | ---: | ---: | ---: | ---: |']
        for c in report['cases']:lines.append(f'| {c["name"]} | {c["fmax_mhz"]:.2f} | {c["logic_lut_cells"]} | {c["flipflops"]} | {c["dsp48"]} |')
        lines += ['', 'LUT counts exclude LUTRAM/SRL; DUT-only synthesis excludes scan fixture. Counts are logical cells, not packed physical LUT sites. Timing includes the fixture. The backend uses the larger 50T fabric, not valid 35T sign-off.', '', '![Observed results](lane-evidence.png)', '']
        (out/'summary.md').write_text('\n'.join(lines))
        (out/'artifact_hashes.json').write_text(json.dumps({f.name:digest(f) for f in out.iterdir() if f.is_file() and f.name!='artifact_hashes.json'},indent=2)+'\n')
        print(out)


if __name__=='__main__':main()
