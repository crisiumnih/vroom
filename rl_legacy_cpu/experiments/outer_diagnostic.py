"""Read-only diagnosis of consumed outer development traces; no new rollouts."""
import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from benchmarks.bldc.run import sha,write_csv
from benchmarks.bldc.current_rl.repeat_study import read,save
from tools.outer_rl.study import verify


def load(path):
    with gzip.open(path,'rt') as f:return np.genfromtxt(f,delimiter=',',names=True)


def run(source,output):
    source=Path(source);output=Path(output);verify(source)
    selection=read(source/'selection.json')
    if sha(source/'selection.json')!=(source/'selection.sha256').read_text().strip():raise ValueError('Selection hash')
    output.mkdir(parents=True,exist_ok=False);rows=[];hashes={};violations=Counter()
    for chosen in selection['rows']:
        arm=source/chosen['arm'];manifest=read(arm/'manifest.json')
        for rel,digest in manifest['artifacts_sha256'].items():
            if sha(arm/rel)!=digest:raise ValueError(f'Changed artifact {arm/rel}')
        directory=arm/'validation'/str(chosen['selected']['step']);metrics=read(directory/'metrics.json')
        for case in metrics:
            for violation in case['violations']:violations[violation.split('@')[0]]+=1
            for branch in ['rl','pi']:
                path=directory/f"{case['case']}_{branch}.csv.gz";hashes[str(path)]=sha(path);t=load(path)
                e=t['reference_rad_s']-t['omega'];en=np.clip(e/25,-1,1)
                speed=4*en**2;current=.5*np.clip(np.hypot(t['i_sd'],t['i_sq'])/4,0,1)**2
                tails=[x for x in case[branch]['events'] if x.get('tail_mean_error_rad_s') is not None]
                settled=sum(x.get('settled_in_observed_window',False) for x in case[branch]['events'])
                rows.append(dict(actor=chosen['label'],seed=chosen['seed'],step=chosen['selected']['step'],
                    case=case['case'],branch=branch,completed=case[branch]['completed'],
                    rmse_rad_s=case[branch]['rmse_rad_s'],max_abs_tail_error_rad_s=max([abs(x['tail_mean_error_rad_s']) for x in tails],default=None),
                    settled_events=settled,event_count=len(case[branch]['events']),
                    mean_cost=float(-t['reward'].mean()),mean_speed_cost=float(speed.mean()),mean_current_cost=float(current.mean()),
                    mean_other_cost=float((-t['reward']-speed-current).mean()),
                    counterfactual_l1_cost=float((-t['reward']-speed+4*np.abs(en)).mean()),
                    current_command_limit_fraction=float(np.mean(np.abs(t['outer_command_a'])>=1.5-1e-6)),
                    voltage_limit_fraction=float(np.mean(t['action_at_limit'])),
                    violations=';'.join(case['violations']) if branch=='rl' else ''))
    write_csv(output/'cases.csv',rows)
    save(output/'diagnosis.json',dict(rows=rows,violation_counts=dict(violations),source_sha256=hashes,
        implementation_sha256=sha(Path(__file__)),numpy=np.__version__,matplotlib=matplotlib.__version__,
        interpretation='Observed trajectories only; reward rescoring does not predict retrained performance',
        error_cost_at_005_rad_s=dict(l2=4*(.05/25)**2,l1=4*(.05/25)),
        zero_error_l2_gradient=0,discount_200ms=.995**200,discount_1s=.995**1000))
    fig,axes=plt.subplots(3,1,figsize=(11,9),layout='constrained')
    for ax,case in zip(axes,['startup','reversal','outer_dev_negative_load']):
        for j,chosen in enumerate(r for r in selection['rows'] if r['seed']==11):
            d=source/chosen['arm']/'validation'/str(chosen['selected']['step'])
            t=load(d/f'{case}_rl.csv.gz');ax.plot(t['time_s'],t['omega'],label=chosen['label'],lw=1.15)
            if j==0:
                b=load(d/f'{case}_pi.csv.gz');ax.plot(b['time_s'],b['omega'],'k--',label='matched PI',lw=1.2)
                ax.plot(t['time_s'],t['reference_rad_s'],color='gray',ls=':',label='reference',lw=1.6)
        ax.set_title(case.replace('_',' '));ax.set_ylabel('Speed (rad/s)');ax.grid(alpha=.2)
    axes[0].legend(ncol=3,fontsize=9);axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Outer capacity study: selected seed-11 trajectories (development only)')
    for suffix in ['png','pdf','svg']:fig.savefig(output/f'speed_diagnosis.{suffix}',dpi=160)
    plt.close(fig)
    pi_cost=np.mean([r['mean_cost'] for r in rows if r['branch']=='pi'])
    rl_cost=np.mean([r['mean_cost'] for r in rows if r['branch']=='rl'])
    lines=['# Outer-controller failure diagnosis','',f'Gate violation counts across48 selected model/case combinations: {dict(violations)}.',
        f'Mean per-case original cost: RL {rl_cost:.6f}, matched PI {pi_cost:.6f}. PI trajectories therefore score better on average under the original objective.',
        'This does not establish objective alignment near the tight settling/tail tolerances, nor prove a learning-algorithm defect.',
        'The 0.05 rad/s tracking error contributes0.000016 under quadratic tracking and0.008 under absolute tracking (same coefficient4 and scale25 rad/s).',
        'A targeted next hypothesis is absolute tracking loss with all other reward terms, gamma, plant, seeds, capacities, scenarios and gates held common between arms.',
        'Gamma=.995 implies about0.2s discount horizon; horizon is a separate unresolved hypothesis and is not changed in the reward study.',
        'These are consumed development cases. No new final tests were evaluated. The figure displays seed11 only for readability; cases.csv retains both seeds.']
    (output/'README.md').write_text('\n\n'.join(lines)+'\n')
    print(json.dumps(dict(violations=dict(violations),rl_mean_cost=rl_cost,pi_mean_cost=pi_cost)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('results/bldc/outer-capacity-v1'));p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.source,a.output)
