"""Paired L2/L1 speed-tracking reward experiment; all other outer settings fixed."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor,as_completed
from contextlib import contextmanager
from unittest.mock import patch
import numpy as np
from stable_baselines3 import DDPG
from benchmarks.bldc.run import ROOT,sha
from benchmarks.bldc.current_rl.repeat_study import read,save,now
from tools.outer_rl import study as original,evaluation
from tools.outer_rl.environment import OuterEnv


class AbsoluteTrackingEnv(OuterEnv):
    def step(self,action):
        obs,reward,terminated,truncated,info=super().step(action)
        delta=[];cost=[]
        for row in info['rows']:
            en=float(np.clip((row['reference_rad_s']-row['omega'])/25.,-1.,1.))
            difference=4.*(abs(en)-en**2)
            row['reward']-=difference;delta.append(difference);cost.append(4.*abs(en))
        info['reward_terms']['speed']=float(np.mean(cost))
        if not terminated:reward-=float(np.mean(delta))
        return obs,float(reward),terminated,truncated,info


@contextmanager
def environment(shape):
    if shape not in ('l2','l1'):raise ValueError('Unknown tracking shape')
    cls=AbsoluteTrackingEnv if shape=='l1' else OuterEnv
    with patch.object(original,'OuterEnv',cls),patch.object(evaluation,'OuterEnv',cls):yield


def prepare(output,source,inner,budget=250000,seeds=(14,15),validation_every=25000):
    output=Path(output).resolve();source=Path(source).resolve()
    original.verify(source)
    if sha(source/'selection.json')!=(source/'selection.sha256').read_text().strip():raise ValueError('Changed capacity selection')
    selected=read(source/'selection.json');label=selected['winner']
    if label!='128x2':raise ValueError('Expected frozen capacity-ranking winner128x2')
    output.mkdir(parents=True,exist_ok=False)
    records=[];children={}
    for shape in ('l2','l1'):
        child=output/shape
        original.prepare(child,inner,budget,seeds,[label],validation_every)
        cfg=read(child/'config.json');cfg['reward']['speed_shape']=shape
        cfg['interpretation']='Paired tracking-reward experiment; original settings unchanged except L2 versus L1 speed term'
        save(child/'config.json',cfg)
        protocol=read(child/'protocol.json');protocol['config']=cfg
        protocol['config_sha256']=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
        protocol['frozen_files']['config.json']=sha(child/'config.json')
        protocol['source_sha256'][str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
        protocol['tracking_shape']=shape;protocol['dispatch']='tools.outer_reward_study run; do not call original CLI for l1'
        save(child/'protocol.json',protocol);(child/'protocol.sha256').write_text(sha(child/'protocol.json')+'\n')
        children[shape]=sha(child/'protocol.json')
        records.extend(dict(shape=shape,label=label,seed=s,arm=f'{label}-seed{s}') for s in seeds)
    p=dict(protocol='outer-tracking-reward-v1',created_utc=now(),source=str(source),
        source_selection_sha256=sha(source/'selection.json'),children=children,records=records,
        seeds=list(seeds),label=label,budget=budget,max_workers=2,max_outer_actions=len(records)*budget,
        max_training_physics_steps=10*len(records)*budget,source_sha256=sha(Path(__file__)),
        selection='Failed seeds, incomplete cases, violations, median speed RMSE, L2 before L1 on exact tie',
        hypothesis='Absolute tracking increases cost sensitivity near settling tolerance; efficacy unproven',
        final_evaluation=False)
    save(output/'protocol.json',p);(output/'protocol.sha256').write_text(sha(output/'protocol.json')+'\n')
    original.atomic(output/'status.json',dict(status='prepared'))
    return p


def verify(output):
    output=Path(output);p=read(output/'protocol.json')
    if sha(output/'protocol.json')!=(output/'protocol.sha256').read_text().strip():raise ValueError('Changed reward protocol')
    if sha(Path(__file__))!=p['source_sha256']:raise ValueError('Changed reward source')
    if sha(Path(p['source'])/'selection.json')!=p['source_selection_sha256']:raise ValueError('Changed parent selection')
    expected={(shape,s) for shape in ('l2','l1') for s in p['seeds']}
    if len(p['records'])!=len(expected) or {(r['shape'],r['seed']) for r in p['records']}!=expected:raise ValueError('Invalid reward queue')
    configs={}
    for shape,digest in p['children'].items():
        if sha(output/shape/'protocol.json')!=digest:raise ValueError('Changed child protocol')
        original.verify(output/shape)
        c=read(output/shape/'config.json')
        if c['reward'].pop('speed_shape')!=shape:raise ValueError('Wrong reward dispatch')
        configs[shape]=c
    if configs['l2']!=configs['l1']:raise ValueError('Uncontrolled configuration difference')
    return p


def run(output,shape,seed):
    output=Path(output);p=verify(output)
    if (shape,seed) not in [(r['shape'],r['seed']) for r in p['records']]:raise ValueError('Undeclared job')
    with environment(shape):original.run_arm(output/shape,p['label'],seed)


def execute(output):
    output=Path(output).resolve();p=verify(output)
    with (output/'queue_started.json').open('x') as f:json.dump(dict(started_utc=now()),f)
    (output/'logs').mkdir(exist_ok=False)
    original.atomic(output/'status.json',dict(status='preparing_baselines'))
    try:
        original.torch.set_num_threads(1);original.torch.use_deterministic_algorithms(True)
        for shape in ('l2','l1'):
            child=output/shape;c=read(child/'config.json');inner=DDPG.load(child/'inner_model.zip',device='cpu')
            with environment(shape):evaluation.baselines(c['plant'],c['inner_study'],inner,c['validation_cases'],child/'baselines')
        original.atomic(output/'status.json',dict(status='running'))
        def job(r):
            started=now()
            try:
                with (output/'logs'/f"{r['shape']}-{r['arm']}.log").open('x') as log:
                    process=subprocess.run([sys.executable,'-m','tools.outer_reward_study','run','--output',str(output),
                        '--shape',r['shape'],'--seed',str(r['seed'])],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                return dict(**r,returncode=process.returncode,started_utc=started,completed_utc=now())
            except Exception as exc:return dict(**r,returncode=-1,failure=f'{type(exc).__name__}: {exc}')
        results=[]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(job,r) for r in p['records']]
            for future in as_completed(futures):
                results.append(future.result());original.atomic(output/'queue_status.json',dict(completed_jobs=len(results),total_jobs=len(p['records']),results=results))
                print(json.dumps(results[-1]),flush=True)
        save(output/'run_summary.json',results)
        if any(r['returncode'] for r in results):raise RuntimeError('Failed job; no aggregate selection')
        select(output)
        original.atomic(output/'status.json',dict(status='completed',completed_utc=now()))
    except BaseException as exc:
        original.atomic(output/'status.json',dict(status='failed',failure=f'{type(exc).__name__}: {exc}',failed_utc=now()))
        raise


def select(output):
    output=Path(output);p=verify(output)
    if (output/'selection.json').exists():raise FileExistsError('Selection already committed')
    queue=read(output/'run_summary.json')
    if len(queue)!=len(p['records']) or {(r['shape'],r['seed']) for r in queue}!={(r['shape'],r['seed']) for r in p['records']} or any(r['returncode'] for r in queue):raise RuntimeError('Incomplete queue')
    rows=[];scores={}
    for i,shape in enumerate(('l2','l1')):
        child=output/shape;save(child/'run_summary.json',[r for r in queue if r['shape']==shape])
        original.select(child)
        selected=read(child/'selection.json')
        for row in selected['rows']:rows.append(dict(shape=shape,**row))
        scores[shape]=selected['scores'][p['label']][:-1]+[i]
        original.atomic(child/'status.json',dict(status='completed',completed_utc=now()))
    result=dict(rows=rows,scores=scores,winner=min(scores,key=scores.get),final_qualification=False,
                all_winner_seeds_pass_validation=min(scores.values())[0]==0)
    save(output/'selection.json',result);(output/'selection.sha256').write_text(sha(output/'selection.json')+'\n')
    lines=['# Outer tracking-reward development comparison','',f"Validation-ranked reward: {result['winner']}. No final qualification.",'',
        '| Reward | Seed | Selected step | Cases passed | Speed RMSE rad/s |','|---|---:|---:|---:|---:|']
    for row in rows:
        s=row['selected'];lines.append(f"| {row['shape']} | {row['seed']} | {s['step']} | {s['passed_cases']}/{s['case_count']} | {s['score'][2]:.6f} |")
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','run','execute','select'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path,default=Path('results/bldc/outer-capacity-v1'))
    p.add_argument('--inner',type=Path,default=Path('results/bldc/outer-inner-freeze-v1'))
    p.add_argument('--shape',choices=['l2','l1']);p.add_argument('--seed',type=int)
    a=p.parse_args()
    if a.command=='prepare':prepare(a.output,a.source,a.inner)
    elif a.command=='run':run(a.output,a.shape,a.seed)
    elif a.command=='execute':execute(a.output)
    else:select(a.output)


if __name__=='__main__':main()
