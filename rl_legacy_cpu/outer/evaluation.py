"""Matched multirate speed PI/RL evaluation; no baseline participates in learning."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from benchmarks.bldc.cascaded import derived_gains,DQCurrentController
from benchmarks.bldc.metrics import summarize
from benchmarks.bldc.current_rl.speed_bridge import gates
from benchmarks.bldc.run import sha,write_trace
from benchmarks.bldc.current_rl.repeat_study import read,save
from .environment import OuterEnv


class SpeedPI:
    def __init__(self,plant):
        self.gains=derived_gains(plant,plant['controller']);self.beta=plant['controller']['speed_setpoint_weight']
        self.limit=plant['controller']['current_reference_limit_a'];self.reset()
    def reset(self):self.integral=0.;self.pending=None
    def act(self,state,reference,dt=.001):
        if dt!=.001:raise ValueError('Outer PI must use1ms')
        error=reference-state['omega']
        raw=self.gains['speed_kp_a_per_rad_s']*(self.beta*reference-state['omega'])+self.integral
        iq=float(np.clip(raw,-self.limit,self.limit));self.pending=(error,raw,iq,dt)
        return np.asarray([iq/self.limit],dtype=np.float32)
    def observe(self,rows):
        if self.pending is None:raise RuntimeError('act required before observe')
        error,raw,iq,dt=self.pending
        increment=self.gains['speed_ki_a_per_rad']*error*min(dt,len(rows)*.0001)
        reference_push=(raw>=self.limit and increment>0) or (raw<=-self.limit and increment<0)
        voltage_push=any(r.get('voltage_saturated',False) and error*(iq-r['i_q_before_a'])>0 for r in rows)
        if not(reference_push or voltage_push):self.integral+=increment
        self.pending=None


def rollout(outer_model,plant,inner_study,inner_model,case,branch):
    if branch not in ('rl','pi','all_pi'):raise ValueError('Unknown controller branch')
    env=OuterEnv(plant,inner_study,inner_model,[case]);rows=[]
    status=dict(seed=9000,completed=False,terminated=False,truncated=False,failure=None,outer_actions=0,physics_steps=0)
    pi=SpeedPI(plant) if branch!='rl' else None
    try:
        observation,_=env.reset(seed=9000,options={'case':case})
        if branch=='all_pi':env.current=DQCurrentController(plant,plant['controller'])
        while True:
            state,reference=env.state_reference()
            action=pi.act(state,reference) if pi else outer_model.predict(observation,deterministic=True)[0]
            observation,_,terminated,truncated,info=env.step(action);rows.extend(info['rows'])
            if pi:pi.observe(info['rows'])
            status.update(outer_actions=info['outer_actions'],physics_steps=info['physics_steps'])
            if terminated or truncated:
                status.update(completed=bool(info['time_limit'] and not terminated),terminated=terminated,
                              truncated=truncated,failure=info['failure'])
                break
    except (RuntimeError,ValueError,FloatingPointError) as exc:
        # Preserve successful inner samples from a numerically failed outer block.
        existing=rows[-1]['time_s'] if rows else -1
        rows.extend(r for r in getattr(env,'last_rows',[]) if r['time_s']>existing)
        status['failure']=f'{type(exc).__name__}: {exc}'
    finally:env.close()
    trace={k:np.asarray([r[k] for r in rows]) for k in rows[0]} if rows else {'time_s':np.asarray([])}
    return trace,summarize(trace,case,plant,status)


def model_digest(model):
    h=hashlib.sha256()
    for name,value in model.policy.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def identity(plant,inner_study,inner_model,cases):
    return dict(plant=plant,inner_study=inner_study,inner_parameters_sha256=model_digest(inner_model),cases=cases,seed=9000)


def baselines(plant,inner_study,inner_model,cases,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False);rows=[]
    save(directory/'identity.json',identity(plant,inner_study,inner_model,cases))
    for case in cases:
        row=dict(case=case['name'])
        for branch in ['pi','all_pi']:
            trace,metrics=rollout(None,plant,inner_study,inner_model,case,branch)
            write_trace(directory/f"{case['name']}_{branch}.csv.gz",trace);row[branch]=metrics
        rows.append(row);save(directory/'metrics.json',rows)
    save(directory/'artifacts.json',{str(f.relative_to(directory)):sha(f) for f in directory.iterdir() if f.is_file()})
    if any(not r[b]['completed'] for r in rows for b in ('pi','all_pi')):raise RuntimeError('Incomplete PI baseline; traces retained')
    return rows


def compare(outer_model,plant,inner_study,inner_model,cases,directory,baseline_directory=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    baseline_directory=Path(baseline_directory) if baseline_directory else directory/'baselines'
    if not baseline_directory.exists():baselines(plant,inner_study,inner_model,cases,baseline_directory)
    if read(baseline_directory/'identity.json')!=identity(plant,inner_study,inner_model,cases):raise ValueError('Changed baseline identity')
    for rel,digest in read(baseline_directory/'artifacts.json').items():
        if sha(baseline_directory/rel)!=digest:raise ValueError('Changed baseline artifact')
    baseline=read(baseline_directory/'metrics.json');rows=[]
    if [r['case'] for r in baseline]!=[c['name'] for c in cases]:raise ValueError('Baseline case mismatch')
    for case,base in zip(cases,baseline):
        if not all(base[b]['completed'] for b in ('pi','all_pi')):raise RuntimeError('Incomplete baseline')
        for branch in ('pi','all_pi'):
            source=baseline_directory/f"{case['name']}_{branch}.csv.gz"
            target=directory/source.name
            try:target.hardlink_to(source)
            except OSError:shutil.copyfile(source,target)
        trace,metrics=rollout(outer_model,plant,inner_study,inner_model,case,'rl')
        write_trace(directory/f"{case['name']}_rl.csv.gz",trace)
        violations,comparable=gates(metrics,base['pi'],4.)
        all_violations,all_comparable=gates(metrics,base['all_pi'],4.)
        rows.append(dict(case=case['name'],pi=base['pi'],all_pi=base['all_pi'],rl=metrics,
            violations=violations,passed=comparable and not violations,
            all_pi_violations=all_violations,all_pi_passed=all_comparable and not all_violations))
        save(directory/'metrics.json',rows)
    return rows
