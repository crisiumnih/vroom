"""Versioned 1-kHz speed environment with frozen 10-kHz RL current control."""
import copy
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from benchmarks.bldc.environment import make_environment, schedule_value
from benchmarks.bldc.current_rl.speed_bridge import RLCurrent


class OuterEnv(gym.Env):
    metadata={'render_modes':[]}
    def __init__(self,plant,inner_study,inner_model,scenarios,config=None):
        super().__init__()
        self.plant=copy.deepcopy(plant);self.inner_study=copy.deepcopy(inner_study)
        self.inner_model=inner_model;self.scenarios=copy.deepcopy(scenarios)
        if config:raise ValueError('v1 outer contract constants are fixed')
        if plant['tau_s']!=.0001 or plant['controller']['current_reference_limit_a']!=1.5:
            raise ValueError('Unsupported inner timing/reference limit')
        self.tau_i=.0001;self.tau_o=.001;self.hold_steps=10
        for c in self.scenarios:
            for t in [c['duration_s']]+[x[0] for x in c['reference']]+[x[0] for x in c['disturbance']]:
                if not np.isclose(t/self.tau_o,round(t/self.tau_o),atol=1e-8,rtol=0):raise ValueError('Scenario off outer grid')
        self.action_space=Box(-1,1,(1,),dtype=np.float32)
        self.observation_space=Box(-1,1,(7,),dtype=np.float32)
        self.env=None;self.done=True;self.current=None

    def state_reference(self):
        state=dict(zip(self.names,np.asarray(self.observation[0])*self.limits))
        return state,float(schedule_value(self.case['reference'],self.physics_steps,self.tau_i))

    def _obs(self):
        s,r=self.state_reference()
        v=np.asarray([s['omega']/25,r/25,(r-s['omega'])/25,s['i_sd']/4,s['i_sq']/4,self.previous_command/1.5,self.z],dtype=np.float32)
        if not np.isfinite(v).all():raise RuntimeError('Nonfinite outer observation')
        return np.clip(v,-1,1)

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);self.close()
        self.case=copy.deepcopy((options or {}).get('case') or self.scenarios[int(self.np_random.integers(len(self.scenarios)))])
        self.env,self.load=make_environment(self.plant,self.case)
        ps=self.env.unwrapped.physical_system;self.names=list(ps.state_names);self.limits=np.asarray(ps.limits)
        self.observation,_=self.env.reset(seed=int(self.np_random.integers(2**31)))
        self.current=RLCurrent(self.plant,self.inner_study,self.inner_model);self.current.reset()
        self.physics_steps=0;self.outer_actions=0;self.z=0.;self.previous_command=0.;self.done=False
        self.last_rows=[]
        return self._obs(),dict(scenario=self.case['name'])

    def step(self,action):
        if self.done:raise RuntimeError('reset required after episode completion')
        a=np.asarray(action,dtype=float).reshape(-1)
        if a.shape!=(1,) or not np.isfinite(a).all():raise ValueError('One finite scalar action required')
        self.last_rows=[]
        command=1.5*float(np.clip(a[0],-1,1));pre,reference=self.state_reference()
        z=self.z;delta=abs(command-self.previous_command)/3.;rows=[];terms=[]
        terminated=False;truncated=False;failure=None
        duration=round(self.case['duration_s']/self.tau_i)
        for _ in range(min(self.hold_steps,duration-self.physics_steps)):
            before,ref=self.state_reference()
            self.load.disturbance_nm=schedule_value(self.case['disturbance'],self.physics_steps,self.tau_i)
            voltage=np.asarray(self.current.act_current(before,[0.,command],self.tau_i),dtype=float)
            if voltage.shape!=(2,) or not np.isfinite(voltage).all():raise RuntimeError('Invalid inner action')
            self.observation,_,native_term,native_trunc,_=self.env.step(voltage)
            after,_=self.state_reference()
            if not np.isfinite(list(after.values())).all():raise RuntimeError('Nonfinite physical state')
            self.physics_steps+=1
            phase_peak=max(abs(after[k]) for k in ['i_a','i_b','i_c'])
            terminated=bool(native_term or phase_peak>4.)
            truncated=bool(native_trunc and not terminated)
            failure=('phase_current_trip' if phase_peak>4 else 'native_constraint') if terminated else ('native_truncation' if truncated else None)
            components=dict(speed=4.*float(np.clip((ref-after['omega'])/25,-1,1))**2,
                current=.5*float(np.clip(np.hypot(after['i_sd'],after['i_sq'])/4,0,1))**2,
                command_delta=.1*float(np.clip(delta,0,1))**2,memory=.5*z*z,
                torque_delta=.1*float(np.clip(abs(after['torque']-before['torque'])/.1,0,1))**2)
            terms.append(components)
            indices=[self.names.index(n) for n in ['i_a','i_b','i_c']]
            row=dict(after,**self.current.diagnostics,time_s=self.physics_steps*self.tau_i,
                reference_rad_s=ref,omega_before=float(before['omega']),disturbance_nm=float(self.load.disturbance_nm),
                requested_d=float(voltage[0]),requested_q=float(voltage[1]),action_d=float(voltage[0]),action_q=float(voltage[1]),
                action_clipped=0.,action_at_limit=float(np.linalg.norm(voltage)>=self.plant['action_norm_limit']-1e-12),
                outer_action=float(a[0]),outer_command_a=command,outer_action_clipped=float(abs(a[0])>1),
                reward=-sum(components.values()),phase_current_ratio=phase_peak/4.,current_constraint_ratio=float(np.sum(np.asarray(self.observation[0])[indices]**2)),
                terminated=float(terminated),truncated=float(truncated))
            rows.append(row);self.last_rows=rows
            if terminated or truncated:break
        if not rows:raise RuntimeError('No inner samples in outer transition')
        self.outer_actions+=1
        self.z=float(np.clip(z+len(rows)*self.tau_i*np.clip((reference-pre['omega'])/25,-1,1)/.5,-1,1))
        self.previous_command=command
        truncated=bool(truncated or (not terminated and self.physics_steps>=duration))
        self.done=terminated or truncated
        reward=-1040. if terminated else -float(np.mean([sum(c.values()) for c in terms]))
        mean_terms={k:float(np.mean([c[k] for c in terms])) for k in terms[0]}
        mean_terms['failure_penalty']=-1040. if terminated else 0.
        info=dict(inner_steps=len(rows),physics_steps=self.physics_steps,outer_actions=self.outer_actions,
            failure=failure,rows=rows,reward_terms=mean_terms,scenario=self.case['name'],
            time_limit=bool(truncated and not failure and self.physics_steps>=duration))
        return self._obs(),reward,bool(terminated),bool(truncated),info

    def close(self):
        if self.env is not None:self.env.close()
        self.env=None;self.current=None;self.done=True
