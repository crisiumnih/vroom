"""Paired discount-horizon experiment with shared L1 tracking and failure penalty."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor,as_completed
from contextlib import contextmanager
from unittest.mock import patch
from stable_baselines3 import DDPG
from benchmarks.bldc.run import ROOT,sha
from benchmarks.bldc.current_rl.repeat_study import read,save,now
from tools.outer_rl import study as original,evaluation
from tools.outer_rl.environment import OuterEnv


from tools import outer_reward_study as reward_study

GAMMAS = {'g0995': .995, 'g0999': .999}
FAILURE_PENALTY = -5200.

class HorizonEnv(reward_study.AbsoluteTrackingEnv):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if terminated:
            reward = FAILURE_PENALTY
            info['reward_terms']['failure_penalty'] = FAILURE_PENALTY
        return obs, reward, terminated, truncated, info

@contextmanager
def environment(group):
    if group not in GAMMAS:
        raise ValueError('Unknown horizon group')
    with patch.object(original, 'OuterEnv', HorizonEnv), patch.object(evaluation, 'OuterEnv', HorizonEnv):
        yield


def prepare(output,source,inner,budget=250000,seeds=(16,17),validation_every=25000):
    output=Path(output).resolve();source=Path(source).resolve()
    reward_study.verify(source)
    if sha(source/'selection.json')!=(source/'selection.sha256').read_text().strip():raise ValueError('Changed reward selection')
    selected=read(source/'selection.json');label='128x2'
    if selected['winner']!='l1':raise ValueError('Expected L1 parent winner')
    output.mkdir(parents=True,exist_ok=False)
    records=[];children={}
    for group in ('g0995','g0999'):
        child=output/group
        original.prepare(child,inner,budget,seeds,[label],validation_every)
        cfg=read(child/'config.json');cfg['reward'].update(speed_shape='l1',failure=FAILURE_PENALTY)
        cfg['learner']['gamma']=GAMMAS[group]
        cfg['interpretation']='Paired discount-horizon experiment; L1 tracking and shared -5200 failure penalty; only gamma differs'
        save(child/'config.json',cfg)
        protocol=read(child/'protocol.json');protocol['config']=cfg
        protocol['config_sha256']=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
        protocol['frozen_files']['config.json']=sha(child/'config.json')
        protocol['source_sha256'][str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
        dependency=Path(reward_study.__file__)
        protocol['source_sha256'][str(dependency.relative_to(ROOT))]=sha(dependency)
        protocol['horizon_group']=group;protocol['dispatch']='tools.outer_horizon_study run; do not call original CLI'
        save(child/'protocol.json',protocol);(child/'protocol.sha256').write_text(sha(child/'protocol.json')+'\n')
        children[group]=sha(child/'protocol.json')
        records.extend(dict(group=group,label=label,seed=s,arm=f'{label}-seed{s}') for s in seeds)
    p=dict(protocol='outer-discount-horizon-v1',created_utc=now(),source=str(source),
        source_selection_sha256=sha(source/'selection.json'),children=children,records=records,
        seeds=list(seeds),label=label,budget=budget,max_workers=2,max_outer_actions=len(records)*budget,
        max_training_physics_steps=10*len(records)*budget,source_sha256=sha(Path(__file__)),
        selection='Failed seeds, incomplete cases, violations, median speed RMSE, gamma .995 before .999 on exact tie',
        hypothesis='Longer discount horizon may reduce sustained offsets; efficacy unproven; shared failure penalty differs from prior reward study',
        final_evaluation=False)
    save(output/'protocol.json',p);(output/'protocol.sha256').write_text(sha(output/'protocol.json')+'\n')
    original.atomic(output/'status.json',dict(status='prepared'))
    return p


def verify(output):
    output=Path(output);p=read(output/'protocol.json')
    if sha(output/'protocol.json')!=(output/'protocol.sha256').read_text().strip():raise ValueError('Changed horizon protocol')
    if sha(Path(__file__))!=p['source_sha256']:raise ValueError('Changed horizon source')
    if sha(Path(p['source'])/'selection.json')!=p['source_selection_sha256']:raise ValueError('Changed parent selection')
    expected={(group,s) for group in ('g0995','g0999') for s in p['seeds']}
    if len(p['records'])!=len(expected) or {(r['group'],r['seed']) for r in p['records']}!=expected:raise ValueError('Invalid horizon queue')
    if set(p['children'])!=set(GAMMAS):raise ValueError('Incomplete horizon groups')
    if any(r['label']!=p['label'] or r['arm']!=f"{p['label']}-seed{r['seed']}" for r in p['records']):raise ValueError('Invalid job identity')
    configs={}
    for group,digest in p['children'].items():
        if sha(output/group/'protocol.json')!=digest:raise ValueError('Changed child protocol')
        original.verify(output/group)
        c=read(output/group/'config.json')
        if c['learner'].pop('gamma')!=GAMMAS[group]:raise ValueError('Wrong gamma dispatch')
        if c['reward']['speed_shape']!='l1' or c['reward']['failure']!=FAILURE_PENALTY:raise ValueError('Wrong shared reward')
        configs[group]=c
    if configs['g0995']!=configs['g0999']:raise ValueError('Uncontrolled configuration difference')
    return p


def run(output,group,seed):
    output=Path(output);p=verify(output)
    if (group,seed) not in [(r['group'],r['seed']) for r in p['records']]:raise ValueError('Undeclared job')
    with environment(group):original.run_arm(output/group,p['label'],seed)


def execute(output):
    output=Path(output).resolve();p=verify(output)
    with (output/'queue_started.json').open('x') as f:json.dump(dict(started_utc=now()),f)
    (output/'logs').mkdir(exist_ok=False)
    original.atomic(output/'status.json',dict(status='preparing_baselines'))
    try:
        original.torch.set_num_threads(1);original.torch.use_deterministic_algorithms(True)
        for group in ('g0995','g0999'):
            child=output/group;c=read(child/'config.json');inner=DDPG.load(child/'inner_model.zip',device='cpu')
            with environment(group):evaluation.baselines(c['plant'],c['inner_study'],inner,c['validation_cases'],child/'baselines')
        original.atomic(output/'status.json',dict(status='running'))
        def job(r):
            started=now()
            try:
                with (output/'logs'/f"{r['group']}-{r['arm']}.log").open('x') as log:
                    process=subprocess.run([sys.executable,'-m','tools.outer_horizon_study','run','--output',str(output),
                        '--group',r['group'],'--seed',str(r['seed'])],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
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
    if len(queue)!=len(p['records']) or {(r['group'],r['seed']) for r in queue}!={(r['group'],r['seed']) for r in p['records']} or any(r['returncode'] for r in queue):raise RuntimeError('Incomplete queue')
    rows=[];scores={}
    for i,group in enumerate(('g0995','g0999')):
        child=output/group;save(child/'run_summary.json',[r for r in queue if r['group']==group])
        original.select(child)
        selected=read(child/'selection.json')
        for row in selected['rows']:rows.append(dict(group=group,**row))
        scores[group]=selected['scores'][p['label']][:-1]+[i]
        original.atomic(child/'status.json',dict(status='completed',completed_utc=now()))
    result=dict(rows=rows,scores=scores,winner=min(scores,key=scores.get),final_qualification=False,
                all_winner_seeds_pass_validation=min(scores.values())[0]==0)
    save(output/'selection.json',result);(output/'selection.sha256').write_text(sha(output/'selection.json')+'\n')
    lines=['# Outer discount-horizon development comparison','',f"Validation-ranked horizon: {result['winner']}. No final qualification.",'',
        '| Horizon | Seed | Selected step | Cases passed | Speed RMSE rad/s |','|---|---:|---:|---:|---:|']
    for row in rows:
        s=row['selected'];lines.append(f"| {row['group']} | {row['seed']} | {s['step']} | {s['passed_cases']}/{s['case_count']} | {s['score'][2]:.6f} |")
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','run','execute','select'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path,default=Path('results/bldc/outer-reward-v1'))
    p.add_argument('--inner',type=Path,default=Path('results/bldc/outer-inner-freeze-v1'))
    p.add_argument('--group',choices=['g0995','g0999']);p.add_argument('--seed',type=int)
    a=p.parse_args()
    if a.command=='prepare':prepare(a.output,a.source,a.inner)
    elif a.command=='run':run(a.output,a.group,a.seed)
    elif a.command=='execute':execute(a.output)
    else:select(a.output)


if __name__=='__main__':main()
