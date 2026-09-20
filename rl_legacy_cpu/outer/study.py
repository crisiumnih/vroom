"""Frozen outer-DDPG capacity screen over a qualified fixed inner policy."""
import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import subprocess
import resource
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from benchmarks.bldc.run import ROOT, provenance, sha
from benchmarks.bldc.current_rl.repeat_study import read, save, now
from benchmarks.bldc.current_rl.speed_bridge import scenarios
from tools.outer_rl.environment import OuterEnv
from tools.outer_rl.evaluation import compare, baselines, model_digest

WIDTHS={'128x2':[128,128],'256x3':[256]*3,'512x3':[512]*3,'1024x3':[1024]*3}


def atomic(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');save(temp,value);temp.replace(path)


def training_cases():
    rng=np.random.default_rng(510)
    cases=[]
    for i in range(40):
        refs=rng.choice([-8.,-6.,-4.,-2.,2.,4.,6.,8.],3).tolist()
        load=float(rng.choice([-.025,0.,.025]))
        cases.append(dict(name=f'train_{i}',duration_s=2.,reference=[[0.,0.],[.1,refs[0]],[.65,refs[1]],[1.25,refs[2]],[1.8,0.]],
            disturbance=[[0.,0.],[.8,load],[1.5,0.]],stress=False))
    return cases


def validation_cases():
    cases=copy.deepcopy(scenarios()[:4])
    cases.extend([
        dict(name='outer_dev_negative',duration_s=1.5,reference=[[0.,0.],[.18,-4.5]],disturbance=[[0.,0.]],stress=False),
        dict(name='outer_dev_negative_load',duration_s=2.5,reference=[[0.,0.],[.18,-5.5]],disturbance=[[0.,0.],[1.,-.025],[1.7,0.]],stress=False)])
    return cases


def verify(output):
    output=Path(output);p=read(output/'protocol.json')
    if sha(output/'protocol.json')!=(output/'protocol.sha256').read_text().strip():raise ValueError('Changed outer protocol')
    for rel,digest in p['source_sha256'].items():
        if sha(ROOT/rel)!=digest:raise ValueError(f'Changed source {rel}')
    for package,version in p['packages'].items():
        if importlib.metadata.version(package)!=version:raise ValueError(f'Changed package {package}')
    for name,digest in p['frozen_files'].items():
        if sha(output/name)!=digest:raise ValueError(f'Changed frozen input {name}')
    return p


def prepare(output,inner,budget=250000,seeds=(10,11),widths=None,validation_every=25000):
    import shutil
    output=Path(output).resolve();inner=Path(inner).resolve();widths=list(widths or WIDTHS)
    if output.exists():raise FileExistsError(output)
    if len(set(widths))!=len(widths) or any(w not in WIDTHS for w in widths):raise ValueError('Invalid widths')
    if not seeds or len(set(seeds))!=len(seeds):raise ValueError('Invalid seeds')
    if validation_every<=0 or budget<=0 or budget%validation_every:raise ValueError('Budget must be positive multiple of validation interval')
    m=read(inner/'manifest.json')
    for name,key in [('inner_model.zip','model_sha256'),('inner_config.json','config_sha256')]:
        if sha(inner/name)!=m[key]:raise ValueError('Changed frozen inner')
    output.mkdir(parents=True)
    for name in ['inner_model.zip','inner_config.json','manifest.json']:
        shutil.copyfile(inner/name,output/('inner_freeze.json' if name=='manifest.json' else name))
    saved=read(output/'inner_config.json')
    saved['plant']['controller']['current_reference_limit_a']=saved['study']['reference_limit_a']
    cfg=dict(protocol='bldc-outer-speed-v1',plant=saved['plant'],inner_study=saved['study'],
        training_cases=training_cases(),validation_cases=validation_cases(),test_cases=[],
        learner=dict(gamma=.995,learning_rate=1e-4,batch_size=128,buffer_size=300000,learning_starts=5000,
                     tau=.005,train_freq=1,gradient_steps=1,noise_sigma=.05,critic_widths=[256,256]),
        contract=dict(version='bldc-outer-speed-v1',outer_dt_s=.001,inner_dt_s=.0001,hold_steps=10,
                      speed_scale_rad_s=25.,current_scale_a=4.,reference_limit_a=1.5,memory_time_s=.5,phase_limit_a=4.,
                      features=['speed','reference','error','id','iq','previous_own_command','bounded_error_memory']),
        reward=dict(speed=4.,current=.5,command_delta=.1,memory=.5,torque_delta=.1,torque_scale_nm=.1,failure=-1040.),
        validation_every=validation_every,total_timesteps=budget,
        interpretation='Scratch outer RL; fixed qualified inner; development screen only; no final evaluation')
    save(output/'config.json',cfg)
    p=provenance(cfg)
    p['packages'].update({n:importlib.metadata.version(n) for n in ['torch','stable-baselines3']})
    for f in sorted((ROOT/'tools/outer_rl').glob('*.py')):p['source_sha256'][str(f.relative_to(ROOT))]=sha(f)
    p.update(protocol='outer-capacity-v1',seeds=list(seeds),widths=widths,max_workers=2,
        budget=budget,max_outer_actions=len(widths)*len(seeds)*budget,
        max_training_physics_steps=10*len(widths)*len(seeds)*budget,
        frozen_files={n:sha(output/n) for n in ['config.json','inner_model.zip','inner_config.json','inner_freeze.json']},
        records=[dict(label=w,seed=s,arm=f'{w}-seed{s}') for w in widths for s in seeds],
        selection='Across seeds: failed seeds, incomplete cases, gate violations, median speed RMSE, declared width order',
        initialization='SB3 default random actor/critic; empty replay; no PI labels or warm start',
        final_evaluation=False)
    save(output/'protocol.json',p);(output/'protocol.sha256').write_text(sha(output/'protocol.json')+'\n')
    atomic(output/'status.json',dict(status='prepared'))
    return p


def score(rows):
    return [sum(not r['rl']['completed'] for r in rows),sum(len(r['violations']) for r in rows),
            float(np.mean([r['rl'].get('rmse_rad_s',float('inf')) for r in rows]))]


class Progress(BaseCallback):
    def __init__(self,output,root,config,inner):
        super().__init__();self.out=output;self.root=root;self.c=config;self.inner=inner
        self.best=None;self.physics=0;self.episodes=0;self.failures=0;self.started=time.perf_counter()
    def _on_step(self):
        info=self.locals['infos'][0];self.physics+=info['inner_steps']
        if info.get('failure'):self.failures+=1
        if self.locals['dones'][0]:
            self.episodes+=1
            with (self.out/'episodes.jsonl').open('a') as f:
                f.write(json.dumps(dict(step=self.num_timesteps,physics_steps=info['physics_steps'],
                    outer_actions=info['outer_actions'],failure=info.get('failure')))+'\n')
        if self.num_timesteps%100==0 or info.get('failure'):
            with (self.out/'physical_samples.jsonl').open('a') as f:
                f.write(json.dumps(dict(step=self.num_timesteps,rows=info['rows'],failure=info.get('failure')))+'\n')
        if self.num_timesteps%self.c['validation_every']==0:
            rows=compare(self.model,self.c['plant'],self.c['inner_study'],self.inner,
                self.c['validation_cases'],self.out/'validation'/str(self.num_timesteps),baseline_directory=self.root/'baselines')
            value=score(rows)
            if not np.isfinite(value[-1]):raise RuntimeError('Nonfinite or missing validation RMSE')
            if self.best is None or value<self.best['score']:
                self.model.save(self.out/'best_model')
                self.best=dict(step=self.num_timesteps,score=value,sha256=sha(self.out/'best_model.zip'),
                    passed_cases=sum(r['passed'] for r in rows),case_count=len(rows))
                atomic(self.out/'selection.json',self.best)
        if self.num_timesteps%1000==0:
            atomic(self.out/'status.json',dict(status='running',outer_actions=self.num_timesteps,
                physics_steps=self.physics,updates=self.model._n_updates,episodes=self.episodes,
                failed_episodes=self.failures,selected=self.best,elapsed_s=time.perf_counter()-self.started))
        return True


def learner(env,widths,seed,cfg):
    c=cfg['learner']
    return DDPG('MlpPolicy',env,seed=seed,device='cpu',learning_rate=c['learning_rate'],
        gamma=c['gamma'],tau=c['tau'],buffer_size=c['buffer_size'],learning_starts=c['learning_starts'],
        batch_size=c['batch_size'],train_freq=c['train_freq'],gradient_steps=c['gradient_steps'],
        action_noise=NormalActionNoise(np.zeros(1),np.ones(1)*c['noise_sigma']),
        policy_kwargs=dict(net_arch=dict(pi=widths,qf=c['critic_widths']),activation_fn=torch.nn.ReLU))


def run_arm(root,label,seed):
    root=Path(root);p=verify(root);cfg=read(root/'config.json')
    if (label,seed) not in [(r['label'],r['seed']) for r in p['records']]:raise ValueError('Undeclared job')
    out=root/f'{label}-seed{seed}';out.mkdir(exist_ok=False)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    inner=DDPG.load(root/'inner_model.zip',device='cpu');inner.policy.set_training_mode(False)
    for parameter in inner.policy.parameters():parameter.requires_grad_(False)
    inner_digest=model_digest(inner)
    env=OuterEnv(cfg['plant'],cfg['inner_study'],inner,cfg['training_cases'])
    started=time.perf_counter();manifest=dict(status='running',label=label,seed=seed,
        protocol_sha256=sha(root/'protocol.json'),inner_model_sha256=sha(root/'inner_model.zip'))
    atomic(out/'manifest.json',manifest)
    try:
        model=learner(env,WIDTHS[label],seed,cfg)
        assert model.replay_buffer.size()==0
        model.save(out/'initial_model')
        manifest.update(initial_replay_size=0,initial_model_sha256=sha(out/'initial_model.zip'),
            actor_parameters=sum(x.numel() for x in model.actor.parameters()))
        atomic(out/'manifest.json',manifest)
        cb=Progress(out,root,cfg,inner)
        model.learn(total_timesteps=cfg['total_timesteps'],callback=cb)
        model.save(out/'last_model');model.save_replay_buffer(out/'replay_buffer.pkl')
        if sha(root/'inner_model.zip')!=p['frozen_files']['inner_model.zip'] or model_digest(inner)!=inner_digest:raise ValueError('Inner checkpoint or parameters changed')
        manifest.update(status='completed',outer_actions=model.num_timesteps,physics_steps=cb.physics,
            gradient_updates=model._n_updates,selected=cb.best,failed_episodes=cb.failures,episodes=cb.episodes,
            final_cases=0,stop_reason='budget_ceiling')
    except BaseException as exc:
        save(out/'failed_transition.json',dict(rows=getattr(env,'last_rows',[]),episode_physics_steps=getattr(env,'physics_steps',None),failure=f'{type(exc).__name__}: {exc}'))
        manifest.update(status='failed',failure=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        env.close();manifest['elapsed_s']=time.perf_counter()-started
        manifest['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        atomic(out/'status.json',dict(status=manifest['status'],elapsed_s=manifest['elapsed_s']))
        manifest['artifacts_sha256']={str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file() and f.name!='manifest.json'}
        atomic(out/'manifest.json',manifest)


def execute(root):
    root=Path(root).resolve();p=verify(root)
    with (root/'queue_started.json').open('x') as f:json.dump(dict(started_utc=now()),f)
    (root/'logs').mkdir(exist_ok=False);atomic(root/'status.json',dict(status='preparing_baselines'))
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    cfg=read(root/'config.json');inner=DDPG.load(root/'inner_model.zip',device='cpu')
    try:
        baselines(cfg['plant'],cfg['inner_study'],inner,cfg['validation_cases'],root/'baselines')
    except BaseException as exc:
        atomic(root/'status.json',dict(status='failed',failure=f'Baseline preparation: {exc}'));raise
    atomic(root/'status.json',dict(status='running'))
    def job(r):
        started=now()
        with (root/'logs'/f"{r['arm']}.log").open('x') as log:
            process=subprocess.run([sys.executable,'-m','tools.outer_rl.study','run','--output',str(root),
                '--label',r['label'],'--seed',str(r['seed'])],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        return dict(**r,returncode=process.returncode,started_utc=started,completed_utc=now())
    results=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(job,r) for r in p['records']]
        for future in as_completed(futures):
            results.append(future.result());atomic(root/'queue_status.json',dict(completed_jobs=len(results),total_jobs=len(p['records']),results=results))
            print(json.dumps(results[-1]),flush=True)
    save(root/'run_summary.json',results)
    if any(r['returncode'] for r in results):
        atomic(root/'status.json',dict(status='failed'));raise RuntimeError('Failed outer job; selection blocked')
    select(root)
    atomic(root/'status.json',dict(status='completed',completed_utc=now()))


def select(root):
    root=Path(root);p=verify(root)
    if (root/'selection.json').exists():raise FileExistsError('Selection already committed')
    queue=read(root/'run_summary.json')
    if len(queue)!=len(p['records']) or {r['arm'] for r in queue}!={r['arm'] for r in p['records']} or any(r['returncode'] for r in queue):raise RuntimeError('Incomplete queue')
    rows=[]
    for r in p['records']:
        out=root/r['arm'];m=read(out/'manifest.json')
        if m['status']!='completed':raise RuntimeError('Incomplete job')
        for rel,digest in m['artifacts_sha256'].items():
            if sha(out/rel)!=digest:raise ValueError('Changed training artifact')
        if sha(out/'best_model.zip')!=m['selected']['sha256']:raise ValueError('Changed selected model')
        selected=m['selected'];cfg=read(root/'config.json')
        if m['outer_actions']!=p['budget'] or selected['step']%cfg['validation_every'] or not 0<selected['step']<=p['budget']:raise ValueError('Invalid step schedule')
        metrics=read(out/'validation'/str(selected['step'])/'metrics.json')
        if [r['case'] for r in metrics]!=[c['name'] for c in cfg['validation_cases']] or score(metrics)!=selected['score']:raise ValueError('Changed selection metrics')
        if sum(r['passed'] for r in metrics)!=selected['passed_cases'] or len(metrics)!=selected['case_count']:raise ValueError('Changed pass count')
        rows.append(dict(**r,**{k:m[k] for k in ['selected','outer_actions','physics_steps','actor_parameters','failed_episodes','elapsed_s']}))
    scores={}
    for i,label in enumerate(p['widths']):
        group=[r['selected'] for r in rows if r['label']==label]
        scores[label]=[sum(x['passed_cases']!=x['case_count'] for x in group),sum(x['score'][0] for x in group),
            sum(x['score'][1] for x in group),float(np.median([x['score'][2] for x in group])),i]
    result=dict(rows=rows,scores=scores,winner=min(scores,key=scores.get),final_qualification=False)
    save(root/'selection.json',result);(root/'selection.sha256').write_text(sha(root/'selection.json')+'\n')
    lines=['# Outer speed RL capacity development screen','','Validation-selected architecture: '+result['winner'],
        'Development selection only. Inner actor is frozen; outer actors start from scratch.','',
        '| Actor | Seed | Outer steps | Physics steps | Selected step | Passed cases | Speed RMSE rad/s |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        s=r['selected'];lines.append(f"| {r['label']} | {r['seed']} | {r['outer_actions']} | {r['physics_steps']} | {s['step']} | {s['passed_cases']}/{s['case_count']} | {s['score'][2]:.6f} |")
    (root/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','run','execute','select'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--inner',type=Path,default=Path('results/bldc/outer-inner-freeze-v1'))
    p.add_argument('--label',choices=list(WIDTHS));p.add_argument('--seed',type=int)
    p.add_argument('--budget',type=int,default=250000);p.add_argument('--validation-every',type=int,default=25000)
    p.add_argument('--seeds',type=int,nargs='+',default=[10,11]);p.add_argument('--widths',nargs='+',choices=list(WIDTHS),default=list(WIDTHS))
    a=p.parse_args()
    if a.command=='prepare':prepare(a.output,a.inner,a.budget,a.seeds,a.widths,a.validation_every)
    elif a.command=='run':run_arm(a.output,a.label,a.seed)
    elif a.command=='execute':execute(a.output)
    else:select(a.output)


if __name__=='__main__':main()
