#!/usr/bin/env python3
"""Numerically verify the iterative tanh unit and retain all development evidence.
No checkpoint export, control integration, hardware connection, or quantization.
Requires only Python/Decimal, Icarus, Verilator; figures use matplotlib.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from model import evaluate,reference,bits,value,TINY,SAT,SHIFTS,FRAC,WIDTH
from constants import block


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def fixtures(seed,count):
    rng=random.Random(seed)
    directed=[0,1,2,0x3fffff,0x7fffff,0x800000,0x800001,0x7f7fffff,
              0x7f800000,0x7f800001,0x7fbfffff,0x7fc00000,0x7fffffff]
    # Both shortcut thresholds, powers of two, and range-reduction boundaries.
    centers=[TINY,SAT,*(bits(2**i) for i in range(-13,4)),
             *(bits(k*math.log(2)/2) for k in range(1,29)),bits(math.atanh(1-2**-25))]
    for center in centers:directed.extend(range(center-16,center+17))
    directed += [bits(10*i/2048) for i in range(2049)]
    samples=directed+[a|0x80000000 for a in directed]
    samples += [rng.getrandbits(32) for _ in range(count)]
    samples += [bits(rng.uniform(-10,10)) for _ in range(count)]
    # Many tiny-but-not-shortcut values exercise cancellation in 1-exp(-2x).
    samples += [rng.randrange(TINY+1,0x3d800000)|(rng.getrandbits(1)<<31) for _ in range(count//4)]
    return samples


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=2805)
    parser.add_argument('--random-count',type=int,default=10000)
    args=parser.parse_args();src=Path(__file__).resolve().parents[1];repo=src.parents[2]
    root=repo/'results/arty_tanh';root.mkdir(parents=True,exist_ok=True)
    out=Path(tempfile.mkdtemp(prefix=time.strftime('cordic-%Y%m%dT%H%M%S-'),dir=root));print(out,flush=True)
    report={'status':'running','scope':'Isolated binary32 tanh development verification, not checkpoint parity',
            'config':vars(args),'python':sys.version,'platform':platform.platform(),'commands':[],
            'internal':{'bits':WIDTH,'fraction_bits':FRAC,'rotation_indices':SHIFTS},
            'accuracy_acceptance_max_ulp':1,'reference_decimal_precisions':[120,180]}
    def run(name,argv,cwd=out):
        start=time.monotonic();entry={'name':name,'argv':[str(a) for a in argv],'cwd':str(cwd)};report['commands'].append(entry)
        try:
            r=subprocess.run(argv,cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=300)
            log,code=r.stdout,r.returncode
        except subprocess.TimeoutExpired as exc:
            log=exc.stdout or b'';log=log.decode(errors='replace') if isinstance(log,bytes) else log;log+='\nTIMEOUT\n';code=124
        (out/(name+'.log')).write_text(log);entry.update(returncode=code,seconds=time.monotonic()-start)
        if code:raise RuntimeError(f'{name} failed ({code}); see {out/(name+".log")}')
        return log
    try:
        for name,cmd in {'git-head':['git','rev-parse','HEAD'],'git-branch':['git','branch','--show-current'],
                         'git-status':['git','status','--short'],'git-submodules':['git','submodule','status'],
                         'iverilog':['iverilog','-V'],'verilator':['verilator','--version']}.items():report[name]=run(name,cmd,repo)
        shutil.copytree(src,out/'source',ignore=shutil.ignore_patterns('__pycache__'))
        report['source_hashes']={str(p.relative_to(out/'source')):digest(p) for p in (out/'source').rglob('*') if p.is_file()}
        rtl=out/'source/rtl/vroom_fp32_tanh.v';text=rtl.read_text()
        assert text.split('    // CONSTANTS_BEGIN\n')[1].split('    // CONSTANTS_END')[0]==block()
        report['constants_verified']=True
        samples=fixtures(args.seed,args.random_count);paths=Counter();ulps=Counter();libm_ulps=Counter();max_latency=0
        with (out/'vectors.hex').open('w') as vf,(out/'oracle.csv').open('w',newline='') as cf:
            writer=csv.writer(cf);writer.writerow(['index','input','integer_model','high_precision_reference','ulp','path','latency'])
            for i,a in enumerate(samples):
                y,invalid,latency,path=evaluate(a);gold,gi=reference(a,120)
                assert (gold,gi)==reference(a,180),(i,hex(a),'reference precision disagreement')
                distance=abs(y-gold);assert invalid==gi and distance<=1,(i,hex(a),hex(y),hex(gold),distance)
                if (a&0x7fffffff)<=0x7f800000:
                    host=bits(math.tanh(struct.unpack('<f',struct.pack('<I',a))[0]));libm_ulps[abs(host-gold)]+=1
                vf.write(f'{a:08x}{y:08x}{gold:08x}{invalid:02x}{latency:04x}\n')
                writer.writerow([i,f'{a:08x}',f'{y:08x}',f'{gold:08x}',distance,path,latency]);paths[path]+=1;ulps[distance]+=1;max_latency=max(max_latency,latency)
        report['oracle']={'vectors':len(samples),'paths':dict(paths),'ulp_histogram':dict(ulps),
                          'native_double_tanh_ulp_histogram':dict(libm_ulps),'max_model_latency':max_latency,
                          'reference_precision_agreement':True}
        print('Oracle checked '+str(len(samples))+' inputs',flush=True)
        run('lint',['verilator','--lint-only','--Wall','--top-module','vroom_fp32_tanh',rtl])
        run('compile',['iverilog','-g2005','-Wall','-s','tb_tanh',f'-Ptb_tanh.COUNT={len(samples)}','-o',out/'test.vvp',rtl,out/'source/sim/tb_tanh.v'])
        log=run('simulate',['vvp',out/'test.vvp',f'+seed={args.seed}'])
        records=[json.loads(line[7:]) for line in log.splitlines() if line.startswith('RESULT ')]
        assert len(records)==1 and records[0]['status']=='passed';report['rtl']=records[0]
        with (out/'responses.csv').open() as f:rows=list(csv.DictReader(f))
        assert len(rows)==len(samples)
        actual_ulps=Counter(abs(int(r['result'],16)-int(r['golden'],16)) for r in rows)
        assert actual_ulps==ulps;report['rtl_ulp_histogram']=dict(actual_ulps)
        # Service rate includes one idle/admission edge after each response.
        assert report['rtl']['worst_case_pair_cycles']==2*max_latency+1
        service=max_latency+1
        report['budget_at_100mhz']={'worst_latency_cycles':max_latency,'worst_service_cycles':service,
            'two_outputs_cycles':2*max_latency+1,'two_outputs_us':(2*max_latency+1)/100,
            'evaluations_per_second':22000,'separate_unit_utilization_percent':22000*service/1000000,
            'scope':'Always-ready sink, no input queue; separate engine may overlap dense computation. Not full scheduling proof.'}
        import matplotlib
        matplotlib.use('Agg');import matplotlib.pyplot as plt
        report['matplotlib_version']=matplotlib.__version__
        fig,axes=plt.subplots(1,2,figsize=(10.5,4.2),layout='constrained')
        finite=[r for r in rows if (int(r['input'],16)&0x7fffffff)<SAT and (int(r['input'],16)&0x7fffffff)>TINY]
        xs=[float(value(int(r['input'],16))) for r in finite]
        ys=[float(value(int(r['result'],16))) for r in finite]
        axes[0].scatter(xs,ys,s=2,alpha=.2);axes[0].set(xlabel='Binary32 input',ylabel='Binary32 output',title='RTL tanh output')
        axes[1].scatter(xs,[int(r['latency']) for r in finite],s=2,alpha=.2);axes[1].set(xlabel='Binary32 input',ylabel='Clock edges to response',title='Measured iterative latency')
        for ax in axes:ax.spines[['top','right']].set_visible(False)
        fig.suptitle(f'CORDIC tanh — {len(rows):,} development inputs; max error {max(actual_ulps)} ULP')
        fig.savefig(out/'tanh.png',dpi=160);fig.savefig(out/'tanh.svg');plt.close(fig)
        report['status']='passed';print('PASS tanh numerical and protocol tests',flush=True)
    except BaseException as exc:report.update(status='failed',error=str(exc));print(str(exc),flush=True)
    finally:
        (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
        lines=['# CORDIC tanh development test','',f'Status: {report["status"]}','','Not checkpoint parity or routed hardware timing.']
        if 'oracle' in report:lines += ['',json.dumps(report['oracle'],indent=2)]
        if 'rtl' in report:lines += ['',json.dumps(report['rtl'],indent=2)]
        if 'budget_at_100mhz' in report:lines += ['',json.dumps(report['budget_at_100mhz'],indent=2)]
        if 'error' in report:lines += ['',report['error']]
        (out/'summary.md').write_text('\n'.join(lines)+'\n')
        (out/'artifact_hashes.json').write_text(json.dumps({str(f.relative_to(out)):digest(f) for f in out.rglob('*') if f.is_file() and f.name!='artifact_hashes.json'},indent=2)+'\n')
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
