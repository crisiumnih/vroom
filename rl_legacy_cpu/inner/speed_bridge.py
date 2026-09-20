"""Development comparison: identical speed PI with PI or frozen RL current loop.

This is a hybrid controller, not full-cascade RL and not a hardware experiment.
"""
import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DDPG

from .. import evaluate as evaluator
from ..cascaded import CascadedPIController, derived_gains
from ..environment import make_environment
from ..run import ROOT, sha, write_trace, write_csv
from .core import CurrentEncoder
from .repeat_study import read, save, now


def scenarios():
    def case(name,duration,refs,loads=None,stress=False):
        return dict(name=name,duration_s=duration,reference=refs,
                    disturbance=loads or [[0.,0.]],stress=stress)
    return [
        case('startup',1.2,[[0.,0.],[.1,5.]]),
        case('positive_steps',2.4,[[0.,0.],[.1,3.],[.8,8.],[1.6,2.]]),
        case('load_release',2.0,[[0.,0.],[.1,5.]],[[0.,0.],[.8,.04],[1.3,0.]]),
        case('reversal',2.0,[[0.,0.],[.1,3.],[.8,-3.],[1.4,0.]],stress=True),
        case('saturation_release',2.5,[[0.,0.],[.1,100.],[1.5,5.]],stress=True),
    ]


class RLCurrent:
    """Frozen direct actor behind the same inner-controller API as DQCurrentController."""
    def __init__(self,plant,study,model):
        self.c=copy.deepcopy(study);self.model=model
        contract=study['contract']
        if contract['version']!='bldc-current-iasa-v1':raise ValueError('Unsupported direct contract')
        if plant['tau_s']!=contract['tau_s'] or plant['action_norm_limit']!=contract['action_norm_limit']:
            raise ValueError('Plant/controller timing or action scale mismatch')
        self.current_limit=plant['controller']['current_reference_limit_a']
        if self.current_limit>study['reference_limit_a']:raise ValueError('Current references exceed training envelope')
        self.gains=derived_gains(plant,plant['controller'])
        self.voltage_scale=plant['supply_v']/2
        self.encoder=CurrentEncoder(contract,'direct');self.reset()

    def reset(self):
        self.encoder.reset();self.diagnostics={}

    def act_current(self,state,reference,dt):
        if dt!=self.c['contract']['tau_s']:raise ValueError('Unexpected sample time')
        ref=np.asarray(reference,dtype=float)
        refnorm=np.linalg.norm(ref)
        target=ref*min(1.,self.current_limit/max(refnorm,1e-30))
        obs=self.encoder.encode(state,target)
        raw=np.asarray(self.model.predict(obs,deterministic=True)[0],dtype=float)
        applied,branches,_=self.encoder.action(raw,state)
        unprojected=branches[:2]*self.c['contract']['action_norm_limit']
        self.diagnostics=dict(i_d_reference_a=float(target[0]),i_q_reference_a=float(target[1]),
            i_d_before_a=float(state['i_sd']),i_q_before_a=float(state['i_sq']),
            current_reference_limited=float(refnorm>self.current_limit+1e-12),
            voltage_saturated=float(np.linalg.norm(branches[:2])>1.+1e-12),
            unsaturated_u_d_v=float(unprojected[0]*self.voltage_scale),
            unsaturated_u_q_v=float(unprojected[1]*self.voltage_scale))
        return applied


class HybridController(CascadedPIController):
    def __init__(self,plant,study,model):
        super().__init__(plant,plant['controller'])
        self.current=RLCurrent(plant,study,model)
        self.reset()
    # act/reset intentionally inherited: there is exactly one outer-PI implementation.


class PhaseCurrentTrip(gym.Wrapper):
    """Post-integration sampled trip; not predictive protection or a hardware clamp."""
    def __init__(self,env,limit):
        super().__init__(env)
        ps=env.unwrapped.physical_system
        self.indices=[ps.state_names.index(n) for n in ('i_a','i_b','i_c')]
        self.scales=np.asarray(ps.limits)[self.indices];self.limit=limit

    def step(self,action):
        obs,reward,terminated,truncated,info=self.env.step(action)
        peak=np.max(np.abs(np.asarray(obs[0])[self.indices]*self.scales))
        return obs,reward,bool(terminated or peak>self.limit),truncated,info


def rollout(plant,case,controller,phase_limit=4.):
    def factory(config,scenario):
        env,load=make_environment(config,scenario)
        return PhaseCurrentTrip(env,phase_limit),load
    with patch.object(evaluator,'make_environment',factory):
        trace,metrics=evaluator.evaluate(plant,case,controller,9000)
    metrics['phase_trip_limit_a']=phase_limit
    # Native GEM normalized constraint and explicit phase-current limit are distinct.
    if len(trace['time_s']):
        peak=np.max(np.abs(np.column_stack([trace[n] for n in ('i_a','i_b','i_c')])),axis=1)
        metrics['phase_limit_violating_samples']=int(np.sum(peak>phase_limit))
    else:metrics['phase_limit_violating_samples']=0
    return trace,metrics


def gates(candidate,baseline,phase_limit):
    """Project development tolerances, frozen before observing speed rollouts."""
    if not baseline['completed']:
        return ['baseline_incomplete'],False
    violations=[]
    if not candidate['completed']:violations.append('incomplete')
    if 'rmse_rad_s' not in candidate:return violations+['no_samples'],True
    if candidate['rmse_rad_s']>1.1*baseline['rmse_rad_s']+.05:violations.append('speed_rmse')
    if candidate['peak_phase_current_a']>min(phase_limit,1.25*baseline['peak_phase_current_a']+.1):
        violations.append('peak_current')
    for r,b in zip(candidate['events'],baseline['events']):
        if b.get('settled_in_observed_window'):
            if not r.get('settled_in_observed_window'):violations.append(f"settling@{b['event_time_s']}")
            elif r['settling_or_recovery_s']>1.5*b['settling_or_recovery_s']+.05:
                violations.append(f"recovery_time@{b['event_time_s']}")
        if b.get('tail_mean_error_rad_s') is not None:
            tail=r.get('tail_mean_error_rad_s')
            if tail is None or abs(tail)>max(.05,1.5*abs(b['tail_mean_error_rad_s'])+.02):
                violations.append(f"tail_speed@{b['event_time_s']}")
    return violations,True


def prepare(output,source):
    output,source=Path(output),Path(source)
    saved=read(source/'config.json');plant=copy.deepcopy(saved['plant']);study=saved['study']
    plant['controller']['current_reference_limit_a']=study['reference_limit_a']
    plant['scenarios']=scenarios()
    output.mkdir(parents=True,exist_ok=False)
    modules=['benchmarks/bldc/current_rl/speed_bridge.py','benchmarks/bldc/current_rl/core.py',
             'benchmarks/bldc/cascaded.py','benchmarks/bldc/environment.py','benchmarks/bldc/evaluate.py',
             'benchmarks/bldc/metrics.py','benchmarks/bldc/interfaces.py']
    # Pin the actual GEM plant implementation as well as the adapters.
    modules+= [str(p.relative_to(ROOT)) for p in sorted((ROOT/'src').rglob('*.py'))]
    spec=dict(protocol='current-speed-bridge-v1',created_utc=now(),plant=plant,study=study,
        phase_limit_a=study['phase_limit_a'],source_config_sha256=sha(source/'config.json'),
        source_sha256={name:sha(ROOT/name) for name in modules},
        packages={n:importlib.metadata.version(n) for n in ('gym_electric_motor','gymnasium','numpy','scipy','torch','stable-baselines3')},
        gates=dict(speed_rmse='<= 1.1*PI + 0.05 rad/s',peak_current='<= min(4 A,1.25*PI+0.1 A)',
                   settling='Where PI settles: <=1.5*PI+0.05 s',tail_error='<=max(0.05,1.5*abs(PI)+0.02) rad/s'),
        representative='Validation-median seed of validation-selected arm, from frozen repeat selection',
        interpretation='Development integration; stress cases outside training envelope; hybrid outer PI retained')
    save(output/'protocol.json',spec)
    (output/'protocol.sha256').write_text(sha(output/'protocol.json')+'\n')


def run(output,repetitions):
    output,repetitions=Path(output),Path(repetitions)
    if sha(output/'protocol.json')!=(output/'protocol.sha256').read_text().strip():raise ValueError('Changed speed protocol')
    spec=read(output/'protocol.json')
    for rel,digest in spec['source_sha256'].items():
        if sha(ROOT/rel)!=digest:raise ValueError(f'Changed speed source: {rel}')
    for name,version in spec['packages'].items():
        if importlib.metadata.version(name)!=version:raise ValueError(f'Changed package: {name}')
    if sha(repetitions/'selection.json')!=(repetitions/'selection.sha256').read_text().strip():raise ValueError('Changed selection')
    selected=read(repetitions/'selection.json');r=selected['representative']
    modelpath=Path(r['parent'])/'direct/best_model.zip'
    if sha(modelpath)!=r['model_sha256']:raise ValueError('Changed model')
    actual=read(Path(r['parent'])/'config.json')
    # Seeds/update count may vary, encoder and plant must not.
    if actual['study']['contract']!=spec['study']['contract']:raise ValueError('Encoder mismatch')
    original=copy.deepcopy(spec['plant']);original['controller']['current_reference_limit_a']=actual['plant']['controller']['current_reference_limit_a']
    original['scenarios']=actual['plant']['scenarios']
    if original!=actual['plant']:raise ValueError('Plant mismatch')
    dest=output/'evaluation';dest.mkdir(exist_ok=False)
    save(dest/'started.json',dict(started_utc=now(),selection_sha256=sha(repetitions/'selection.json'),
        representative=r,protocol_sha256=sha(output/'protocol.json')))
    torch.set_num_threads(1);model=DDPG.load(modelpath,device='cpu')
    plant=spec['plant'];rows=[]
    for case in plant['scenarios']:
        pair={}
        for name,controller in [('pi',CascadedPIController(plant,plant['controller'])),
                                ('rl',HybridController(plant,spec['study'],model))]:
            trace,metrics=rollout(plant,case,controller,spec['phase_limit_a'])
            write_trace(dest/f"{case['name']}_{name}.csv.gz",trace);pair[name]=metrics
        violations,comparable=gates(pair['rl'],pair['pi'],spec['phase_limit_a'])
        rows.append(dict(case=case['name'],stress=case['stress'],**pair,violations=violations,
                         comparable=comparable,passed=comparable and not violations))
        save(dest/'metrics.json',rows)
    result=dict(completed_utc=now(),representative=r,rows=rows,
        nominal_passed=sum(x['passed'] for x in rows if not x['stress']),
        nominal_cases=sum(not x['stress'] for x in rows),
        stress_passed=sum(x['passed'] for x in rows if x['stress']),
        no_retraining=True,interpretation=spec['interpretation'])
    save(dest/'summary.json',result)
    save(dest/'artifacts.json',{str(p.relative_to(dest)):sha(p) for p in dest.iterdir() if p.is_file()})
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','run']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path,default=Path('results/bldc/current-reconciled-long-v1/direct_duration'))
    p.add_argument('--repetitions',type=Path,default=Path('results/bldc/current-direct-repeat-v1'))
    a=p.parse_args()
    if a.command=='prepare':prepare(a.output,a.source)
    else:run(a.output,a.repetitions)
