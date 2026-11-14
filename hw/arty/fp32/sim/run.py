#!/usr/bin/env python3
"""Verify real FP32 RTL against an exact integer oracle and native C cross-check.
Retains input/source snapshots, full transfer CSVs, bounded VCD windows, logs,
versions, configuration and hashes. Does not touch the baseline dense core.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time
from reference import generate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,default=2805)
    p.add_argument('--random-pairs',type=int,default=50000)
    args=p.parse_args()
    src=Path(__file__).resolve().parents[1];repo=src.parents[2]
    root=repo/'results/arty_fp32';root.mkdir(parents=True,exist_ok=True)
    out=Path(tempfile.mkdtemp(prefix=time.strftime('lane-%Y%m%dT%H%M%S-'),dir=root))
    print(out,flush=True)
    report={'status':'running','config':vars(args),'python':platform.python_version(),
            'platform':platform.platform(),'commands':[],'checks':[],'sources':{},
            'scope':'Real FP32 RTL numerical/protocol simulation; no hardware timing claim'}
    def run(name,argv,cwd=out):
        started=time.monotonic();record={'name':name,'argv':[str(x) for x in argv],'cwd':str(cwd)}
        report['commands'].append(record)
        try:
            r=subprocess.run(argv,cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=240)
            log,code=r.stdout,r.returncode
        except subprocess.TimeoutExpired as e:
            log=e.stdout or b'';log=log.decode(errors='replace') if isinstance(log,bytes) else log
            log+='\nTIMEOUT\n';code=124
        (out/(name+'.log')).write_text(log)
        record.update(returncode=code,seconds=time.monotonic()-started)
        if code:raise RuntimeError(f'{name} failed ({code}); see {out/(name+".log")}')
        return log
    try:
        for name,cmd in {'git-head':['git','rev-parse','HEAD'],'git-branch':['git','branch','--show-current'],
                         'git-status':['git','status','--short'],'git-submodules':['git','submodule','status'],
                         'iverilog':['iverilog','-V'],'verilator':['verilator','--version'],'cc':['cc','--version']}.items():
            report[name]=run(name,cmd,repo)
        snapshot=out/'source';shutil.copytree(src,snapshot,ignore=shutil.ignore_patterns('__pycache__'))
        vendor=out/'hardfloat';shutil.copytree(repo/'third_party/hardfloat',vendor)
        for folder in (snapshot,vendor):
            for f in folder.rglob('*'):
                if f.is_file():report['sources'][str(f.relative_to(out))]=digest(f)
        fixtures=generate(out,args.seed,args.random_pairs);report['fixtures']=fixtures
        run('native-build',['cc','-std=c11','-O0','-ffp-contract=off','-fno-fast-math','-frounding-math',
                            snapshot/'sim/native_reference.c','-o',out/'native-reference','-lm'])
        run('native-oracle',[out/'native-reference',out/'operators.txt'])
        report['checks'].append({'name':'native-oracle','status':'passed','operations':2*fixtures['operator_pairs']})
        rtl=sorted((snapshot/'rtl').glob('*.v'))
        lib=sorted((vendor/'source').glob('*.v'))+[vendor/'source/RISCV/HardFloat_specialize.v']
        inc=['-I'+str(vendor/'source'),'-I'+str(vendor/'source/RISCV')]
        for top in ['vroom_fp32_add','vroom_fp32_mul','vroom_fp32_lane']:
            run('lint-'+top,['verilator','--lint-only','--Wall','--top-module',top,snapshot/'sim/vendor_lint.vlt',*inc,*lib,*rtl])
            report['checks'].append({'name':'lint-'+top,'status':'passed','vendor_only_waivers':True})
        cases=[('operators','tb_fp32_ops',fixtures['operator_pairs'],None)]
        cases += [(f'lane{c["groups"]}','tb_fp32_lane',c['operations'],c['groups']) for c in fixtures['lane_cases']]
        for name,top,count,groups in cases:
            work=out/name;work.mkdir()
            for f in out.glob('*.hex'):shutil.copy2(f,work/f.name)
            exe=work/'test.vvp'
            argv=['iverilog','-g2005','-Wall','-s',top,'-o',exe,f'-P{top}.COUNT={count}',*inc,*lib,*rtl,snapshot/f'sim/{top}.v']
            if groups is not None:argv.insert(1,f'-P{top}.GROUPS={groups}')
            run(name+'-build',argv,work)
            log=run(name,['vvp',exe,f'+seed={args.seed}',*([f'+vectors=lane{groups}.hex'] if groups else [])],work)
            records=[json.loads(line[7:]) for line in log.splitlines() if line.startswith('RESULT ')]
            assert len(records)==1 and records[0]['status']=='passed'
            report['checks'].append({'name':name,**records[0]});print('PASS '+name,flush=True)
        report['status']='passed'
    except BaseException as error:
        report.update(status='failed',error=str(error));print(str(error),flush=True)
    finally:
        (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
        lines=[f'Status: {report["status"]}',report['scope']]
        lines += [f'{c["name"]}: {c["status"]}' for c in report['checks']]
        if 'error' in report:lines.append(report['error'])
        (out/'summary.txt').write_text('\n'.join(lines)+'\n')
        (out/'artifact_hashes.json').write_text(json.dumps({str(f.relative_to(out)):digest(f) for f in out.rglob('*') if f.is_file() and f.name!='artifact_hashes.json'},indent=2)+'\n')
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
