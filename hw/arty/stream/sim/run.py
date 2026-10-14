#!/usr/bin/env python3
"""Exercise complete three-layer actor RTL with real FP32 and CORDIC arithmetic.
Synthetic weights and prepared observations only; no checkpoint/motor/DDR access.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from vectors import generate


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,default=2805)
    p.add_argument('--lanes',nargs='+',type=int,choices=[12,16],default=[12,16])
    args=p.parse_args();arty=Path(__file__).resolve().parents[2];repo=arty.parents[1]
    root=repo/'results/arty_stream';root.mkdir(parents=True,exist_ok=True)
    out=Path(tempfile.mkdtemp(prefix=time.strftime('inference-%Y%m%dT%H%M%S-'),dir=root));print(out,flush=True)
    report={'status':'running','scope':'Complete synthetic actor inference RTL; no DDR, concurrent scheduling or hardware timing claim',
            'config':vars(args),'python':sys.version,'platform':platform.platform(),'commands':[],'checks':[],
            'excluded':['DDR and cache routing','concurrent inner/outer scheduling','checkpoint numerical/trajectory parity','physical timing and device fit'],
            'full_deadline_proven':False}
    def run(name,argv,cwd=out,timeout=300):
        start=time.monotonic();record={'name':name,'argv':[str(a) for a in argv],'cwd':str(cwd)};report['commands'].append(record)
        try:
            r=subprocess.run(argv,cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout)
            code,log=r.returncode,r.stdout
        except subprocess.TimeoutExpired as exc:
            code=124;log=exc.stdout or b'';log=log.decode(errors='replace') if isinstance(log,bytes) else log;log+='\nTIMEOUT\n'
        (out/(name+'.log')).write_text(log);record.update(returncode=code,seconds=time.monotonic()-start)
        if code:raise RuntimeError(f'{name} failed ({code}); see {out/(name+".log")}')
        return log
    try:
        for name,cmd in {'git-head':['git','rev-parse','HEAD'],'git-branch':['git','branch','--show-current'],
                         'git-status':['git','status','--short'],'git-submodules':['git','submodule','status'],
                         'iverilog':['iverilog','-V'],'verilator':['verilator','--version']}.items():report[name]=run(name,cmd,repo)
        source=out/'source';source.mkdir()
        for name in ['stream','fp32','tanh']:shutil.copytree(arty/name,source/name,ignore=shutil.ignore_patterns('__pycache__'))
        vendor=out/'hardfloat';shutil.copytree(repo/'third_party/hardfloat',vendor)
        report['source_hashes']={str(f.relative_to(out)):digest(f) for folder in [source,vendor] for f in folder.rglob('*') if f.is_file()}
        report['fixtures']=generate(out,source,args.seed)
        rtl=[*sorted((source/'stream/rtl').glob('*.v')),*sorted((source/'fp32/rtl').glob('*.v')),*sorted((source/'tanh/rtl').glob('*.v'))]
        lib=[*sorted((vendor/'source').glob('*.v')),vendor/'source/RISCV/HardFloat_specialize.v']
        inc=['-I'+str(vendor/'source'),'-I'+str(vendor/'source/RISCV')]
        for lanes in args.lanes:
            name=f'lanes{lanes}';work=out/name;work.mkdir()
            for f in out.glob('*.hex'):shutil.copy2(f,work/f.name)
            run(name+'-lint',['verilator','--lint-only','--Wall','--top-module','vroom_actor_stream',f'-GLANES={lanes}',source/'fp32/sim/vendor_lint.vlt',*inc,*lib,*rtl])
            run(name+'-compile',['iverilog','-g2005','-Wall','-s','tb_actor_stream',f'-Ptb_actor_stream.LANES={lanes}','-o',work/'test.vvp',*inc,*lib,*rtl,source/'stream/sim/tb_actor_stream.v'])
            log=run(name+'-simulate',['vvp',work/'test.vvp',f'+seed={args.seed}'],work)
            records=[json.loads(line[7:]) for line in log.splitlines() if line.startswith('RESULT ')]
            assert len(records)==1 and records[0]['status']=='passed'
            with (work/'jobs.csv').open() as f:jobs=list(csv.DictReader(f))
            for job in jobs:
                for field in ['job','actor','mode','accepted_edge','first_output_edge','latency','output_stall_cycles']:job[field]=int(job[field])
                job['nominal_us_at_100mhz']=job['latency']/100
                expected=report['fixtures'][job['job']]
                assert [job['output0'],job['output1']]==expected['outputs']
            report['checks'].append({**records[0],'jobs_measured':jobs,'strict_lint':'passed'})
            print('PASS '+name+': '+', '.join(f"job {j['job']}={j['latency']} cycles" for j in jobs),flush=True)
        if len(report['checks'])==2:
            assert [(j['output0'],j['output1']) for j in report['checks'][0]['jobs_measured']]==[(j['output0'],j['output1']) for j in report['checks'][1]['jobs_measured']]
            report['cross_lane_bit_parity']=True
        import matplotlib
        matplotlib.use('Agg');import matplotlib.pyplot as plt
        report['matplotlib_version']=matplotlib.__version__
        fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
        width=.8/len(report['checks'])
        for i,c in enumerate(report['checks']):
            ys=[j['nominal_us_at_100mhz'] for j in c['jobs_measured']]
            ax.bar([j+i*width for j in range(4)],ys,width=width,label=f"{c['lanes']} lanes")
        ax.axhline(100,color='black',ls='--',label='Inner period; outer period = 1000 us')
        ax.set(xticks=[j+(len(report['checks'])-1)*width/2 for j in range(4)],xticklabels=['Inner\nready source','Outer\nready source','Inner\nsource bubbles','Outer\nsource bubbles'],ylabel='Job accept to output-valid (us at assumed 100 MHz)',title='Complete synthetic FP32 inference; physical clock/DDR not proven')
        ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=8)
        fig.savefig(out/'inference.png',dpi=160);fig.savefig(out/'inference.svg');plt.close(fig)
        report['status']='passed'
    except BaseException as exc:report.update(status='failed',error=str(exc));print(str(exc),flush=True)
    finally:
        (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
        lines=['# Complete actor stream verification','',f'Status: {report["status"]}','',report['scope'],'',
               '| Lanes | Job | Actor | Source | Clocks | Nominal us at 100 MHz |', '| --- | --- | --- | --- | ---: | ---: |']
        for c in report['checks']:
            for j in c['jobs_measured']:lines.append(f"| {c['lanes']} | {j['job']} | {'outer' if j['actor'] else 'inner'} | {'bubbles' if j['mode'] else 'ready'} | {j['latency']} | {j['nominal_us_at_100mhz']:.2f} |")
        if 'error' in report:lines+=['',report['error']]
        (out/'summary.md').write_text('\n'.join(lines)+'\n')
        (out/'artifact_hashes.json').write_text(json.dumps({str(f.relative_to(out)):digest(f) for f in out.rglob('*') if f.is_file() and f.name!='artifact_hashes.json'},indent=2)+'\n')
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
