"""Current tracking environment, causal actor memory and physical evaluation."""
import copy
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from gym_electric_motor.physical_systems import ConstantSpeedLoad
from ..environment import make_environment
from ..cascaded import DQCurrentController
from .filter import CurrentFilter


class CurrentEncoder:
    def __init__(self, config, mode):
        self.c=config; self.mode=mode; self.reset()

    def reset(self):
        self.z=np.zeros(2);self.previous=np.zeros(4);self.history=np.zeros((self.c['history'],2))

    def encode(self,state,ref):
        i=np.array([state['i_sd'],state['i_sq']]); s=self.c['current_scale_a']
        values=np.r_[i/s,np.asarray(ref)/s,(np.asarray(ref)-i)/(2*s),self.previous,
                     self.history.ravel()/s,state['omega']/self.c['speed_scale_rad_s'],
                     np.sin(state['epsilon']),np.cos(state['epsilon']),self.z/self.c['integrator_bound']]
        if not np.isfinite(values).all():raise ValueError('Nonfinite current observation')
        return np.clip(values,-1,1).astype(np.float32)

    def action(self,raw,state,filter_=None):
        raw=np.asarray(raw,dtype=float)
        size=2 if self.mode=='direct' else 4
        if raw.shape!=(size,) or not np.isfinite(raw).all():raise ValueError('Invalid actor output')
        raw=np.clip(raw,-1,1)
        branches=np.r_[raw,np.zeros(2)] if size==2 else raw.copy()
        if size==4:self.z+=self.c['integrator_gain']*raw[2:]
        combined=raw[:2]+self.z
        requested=combined*self.c['action_norm_limit']
        applied=requested*min(1,self.c['action_norm_limit']/max(np.linalg.norm(requested),1e-30))
        diagnostics=dict(intervened=False,predicted_peak_a=None,correction_norm=0.)
        if filter_ is not None:applied,diagnostics=filter_.project(state,requested)
        if size==4:
            self.z+=self.c['antiwindup_gain']*(applied/self.c['action_norm_limit']-combined)
            self.z=np.clip(self.z,-self.c['integrator_bound'],self.c['integrator_bound'])
        self.previous=branches
        if len(self.history):
            self.history=np.roll(self.history,-1,axis=0)
            self.history[-1]=[state['i_sd'],state['i_sq']]
        return applied,branches,diagnostics


def task_reward(error,branches,mode,step,config,failed):
    gamma=config['gamma']; s=config['contract']['current_scale_a']
    tracking=-(1-gamma)*float(np.mean(np.sqrt(np.minimum(np.abs(error)/s,1))))
    kp=ki=0.
    if mode=='iasa_sec':
        total=config['total_timesteps']
        kp=config['sec']['kappa_p']*min(1,max(0,(total-step)/(total-config['sec']['decay_p_start'])))
        ki=config['sec']['kappa_i']*min(1,max(0,(total-step)/(total-config['sec']['decay_i_start'])))
    p=-(1-gamma)*kp*float(np.mean(np.sqrt(np.abs(branches[:2]))))
    i=-(1-gamma)*ki*float(np.mean(np.sqrt(np.abs(branches[2:]))))
    terms=dict(tracking=tracking/(1+kp+ki),sec_p=p/(1+kp+ki),sec_i=i/(1+kp+ki),terminal=-1. if failed else 0.)
    return sum(terms.values()),terms


class CurrentEnv(gym.Env):
    metadata={'render_modes':[]}
    def __init__(self,plant,config,mode):
        self.plant=copy.deepcopy(plant);self.config=copy.deepcopy(config);self.mode=mode
        self.encoder=CurrentEncoder(config['contract'],mode)
        self.action_space=Box(-1,1,(2 if mode=='direct' else 4,),dtype=np.float32)
        self.observation_space=Box(-1,1,(13+2*config['contract']['history']+2,),dtype=np.float32)
        self.env=None;self.done=True;self.total_steps=0

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);self.close(); options=options or {}
        self.case=options.get('case')
        speed=self.case['speed_rad_s'] if self.case else float(self.np_random.choice(self.config['training_speeds']))
        scenario=dict(reference=[[0,speed]],duration_s=self.config['episode_s'])
        load=ConstantSpeedLoad(omega_fixed=speed,load_initializer={'states':{'omega':speed}})
        self.env,_=make_environment(self.plant,scenario,load)
        ps=self.env.unwrapped.physical_system;self.names=list(ps.state_names);self.limits=np.asarray(ps.limits)
        self.observation,_=self.env.reset(seed=int(self.np_random.integers(2**31)))
        self.encoder.reset();self.k=0;self.done=False;self.ref=self.reference()
        return self.encoder.encode(self.state(),self.ref),{}

    def state(self):return dict(zip(self.names,np.asarray(self.observation[0])*self.limits))

    def reference(self):
        dt=self.plant['tau_s']
        if self.case:
            return np.array([v for t,v in self.case['reference'] if round(t/dt)<=self.k][-1],dtype=float)
        if self.k % round(self.config['reference_hold_s']/dt)==0:
            if self.k==0 and self.np_random.random()<.25:return np.zeros(2)
            return self.np_random.uniform(-self.config['reference_limit_a'],self.config['reference_limit_a'],2)
        return self.ref

    def step(self,raw):
        if self.done:raise RuntimeError('Reset required')
        before=self.state();ref=self.ref.copy()
        applied,branches,diag=self.encoder.action(raw,before)
        return self.step_voltage(applied,branches,diag,ref)

    def step_voltage(self,applied,branches=None,diag=None,ref=None):
        if self.done:raise RuntimeError('Reset required')
        ref=self.ref.copy() if ref is None else ref
        self.observation,_,terminated,truncated,_=self.env.step(applied)
        after=self.state()
        if not np.isfinite(list(after.values())).all():raise RuntimeError('Nonfinite plant state')
        peak=max(abs(after[n]) for n in ('i_a','i_b','i_c'))
        terminated=bool(terminated or peak>self.config['phase_limit_a'])
        error=ref-np.array([after['i_sd'],after['i_sq']])
        reward,terms=task_reward(error,np.zeros(4) if branches is None else branches,self.mode,self.total_steps,self.config,terminated)
        self.k+=1;self.total_steps+=1
        duration=self.case['duration_s'] if self.case else self.config['episode_s']
        truncated=bool(truncated or (not terminated and self.k>=round(duration/self.plant['tau_s'])))
        self.done=terminated or truncated
        info=dict(state=after,reference=ref.tolist(),action=np.asarray(applied).tolist(),reward_terms=terms,
                  failure='current_constraint' if terminated else None,filter=diag or {})
        self.ref=self.reference()
        return self.encoder.encode(after,self.ref),float(reward),terminated,truncated,info

    def close(self):
        if self.env is not None:self.env.close();self.env=None
        self.done=True


def rollout(plant,config,case,mode='direct',model=None,filtered=False):
    env=CurrentEnv(plant,config,mode);obs,_=env.reset(seed=9000,options={'case':case})
    pi=DQCurrentController(plant,plant['controller']) if model is None else None
    filter_=CurrentFilter(plant,config['phase_limit_a'],config['filter_margin_a']) if filtered else None
    rows=[]
    try:
        while not env.done:
            ref=env.ref.copy();state=env.state()
            if pi is not None:
                applied=pi.act_current(state,ref,plant['tau_s']);branches=np.zeros(4);diag={}
                if filter_:applied,diag=filter_.project(state,applied)
            else:
                raw=model.predict(obs,deterministic=True)[0]
                applied,branches,diag=env.encoder.action(raw,state,filter_)
            obs,_,term,trunc,info=env.step_voltage(applied,branches,diag,ref)
            row=dict(info['state'],time_s=env.k*plant['tau_s'],reference_d_a=ref[0],reference_q_a=ref[1],
                     action_d=applied[0],action_q=applied[1],intervened=float(diag.get('intervened',False)))
            rows.append(row)
        trace={n:np.array([r[n] for r in rows]) for n in rows[0]}
        return trace,metrics(trace,case,not term,config)
    finally:env.close()


def metrics(trace,case,completed,config):
    error=np.column_stack([trace['reference_d_a']-trace['i_sd'],trace['reference_q_a']-trace['i_sq']])
    events=[];dt=config['contract']['tau_s']
    for j,(start,ref) in enumerate(case['reference']):
        end=case['reference'][j+1][0] if j+1<len(case['reference']) else case['duration_s']
        select=(trace['time_s']>start+1e-10)&(trace['time_s']<=end+1e-10)
        tail=select&(trace['time_s']>end-.01+1e-10)
        # Full planned event must be observed; partial tails cannot pass.
        reached=bool(len(trace['time_s']) and trace['time_s'][-1]>=end-1e-10)
        e=error[select];t=trace['time_s'][select]-start
        outside=np.flatnonzero(np.max(np.abs(e),axis=1)>config['settling_band_a']) if len(e) else np.array([])
        index=int(outside[-1]+1) if len(outside) else 0
        settle=float(t[index]) if reached and index<len(t) and end-start-t[index]>=config['settling_hold_s']-1e-10 else None
        events.append(dict(start_s=start,end_s=end,reached=reached,settling_s=settle,
            tail_bias_a=np.mean(error[tail],axis=0).tolist() if reached and tail.any() else None,
            tail_rmse_a=float(np.sqrt(np.mean(np.sum(error[tail]**2,axis=1)))) if reached and tail.any() else None))
    return dict(completed=completed,rmse_a=float(np.sqrt(np.mean(np.sum(error**2,axis=1)))),
        peak_phase_a=float(max(np.max(np.abs(trace[n])) for n in ('i_a','i_b','i_c'))),
        intervention_fraction=float(np.mean(trace['intervened'])),events=events)


def gate(result,baseline,config):
    v=[]
    if not result['completed']:v.append('incomplete')
    if result['rmse_a']>1.1*baseline['rmse_a']+.02:v.append('rmse')
    if result['peak_phase_a']>min(config['phase_limit_a'],1.25*baseline['peak_phase_a']+.1):v.append('current')
    for r,b in zip(result['events'],baseline['events']):
        if b['settling_s'] is not None and r['settling_s'] is None:v.append(f"settling@{r['start_s']}")
        if r['tail_rmse_a'] is None or r['tail_rmse_a']>max(.05,1.5*b['tail_rmse_a']+.02):v.append(f"tail@{r['start_s']}")
    return v
