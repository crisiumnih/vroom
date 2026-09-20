"""Post-hoc L1 trace/memory diagnosis, without new policy rollouts."""
import argparse,gzip
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from benchmarks.bldc.run import sha,write_csv
from benchmarks.bldc.current_rl.repeat_study import read,save
from tools.outer_reward_study import verify


def load(path):
    with gzip.open(path,'rt') as f:return np.genfromtxt(f,delimiter=',',names=True)


def memory(trace):
    block=trace[::10];z=0.;out=[]
    for row in block:
        out.append(z)
        z=float(np.clip(z+.001*np.clip((row['reference_rad_s']-row['omega_before'])/25.,-1,1)/.5,-1,1))
    return np.asarray(out)


def command_stats(q):
    q=np.asarray(q,float);x=q-q.mean();power=abs(np.fft.rfft(x))**2;freq=np.fft.rfftfreq(len(x),.001)
    return dict(std_a=float(np.std(q)),peak_to_peak_a=float(np.ptp(q)),
        lag_one_correlation=float(np.corrcoef(q[:-1],q[1:])[0,1]) if np.std(q[:-1])>0 and np.std(q[1:])>0 else None,
        high_frequency_power_fraction=float(np.sum(power[freq>=250])/np.sum(power)) if np.sum(power)>0 else None)


def run(source,output):
    source=Path(source);output=Path(output);verify(source)
    if sha(source/'selection.json')!=(source/'selection.sha256').read_text().strip():raise ValueError('Changed selection')
    selection=read(source/'selection.json');rows=[];hashes={};traces={}
    output.mkdir(parents=True,exist_ok=False)
    for selected in selection['rows']:
        if selected['shape']!='l1':continue
        arm=source/'l1'/selected['arm'];manifest=read(arm/'manifest.json')
        for name,digest in manifest['artifacts_sha256'].items():
            if sha(arm/name)!=digest:raise ValueError('Changed learning artifact')
        folder=arm/'validation'/str(selected['selected']['step'])
        for case in read(folder/'metrics.json'):
            path=folder/f"{case['case']}_rl.csv.gz";hashes[str(path)]=sha(path);t=load(path);block=t[::10];z=memory(t)
            tail=block[-200:];error=tail['reference_rad_s']-tail['omega']
            events=[e for e in case['rl']['events'] if e.get('tail_mean_error_rad_s') is not None]
            rows.append(dict(seed=selected['seed'],step=selected['selected']['step'],case=case['case'],
                final_200ms_bias_rad_s=float(error.mean()),final_200ms_error_std_rad_s=float(error.std()),
                max_abs_event_tail_bias_rad_s=max(abs(e['tail_mean_error_rad_s']) for e in events),
                z_min=float(z.min()),z_max=float(z.max()),z_bound_fraction=float(np.mean(abs(z)>=1-1e-12)),
                **command_stats(tail['outer_command_a']),violations=';'.join(case['violations'])))
            traces[(selected['seed'],case['case'])]=(t,block,z)
            if selected['seed']==14 and case['case']=='outer_dev_negative_load':
                pi_path=folder/f"{case['case']}_pi.csv.gz";pi=load(pi_path);hashes[str(pi_path)]=sha(pi_path)
    write_csv(output/'cases.csv',rows)
    save(output/'diagnosis.json',dict(rows=rows,source_sha256=hashes,implementation_sha256=sha(Path(__file__)),
        software=dict(numpy=np.__version__,matplotlib=matplotlib.__version__),
        windows='Final200 outer samples (0.2 s); FFT mean-centered command, >=250 Hz band, sampling1kHz',
        null_semantics='Undefined lag correlation or zero AC-power fraction is null',
        interpretation='Memory reconstructed exactly from pre-action error; chatter/offset measured, gamma causality unproven'))
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for seed in [14,15]:
        t,b,z=traces[(seed,'outer_dev_negative_load')]
        axes[0,0].plot(t['time_s'],t['omega'],label=f'L1 seed{seed}')
        axes[0,1].plot(b['time_s'],z,label=f'L1 seed{seed}')
        for ax,case,start,end in [(axes[1,0],'startup',1.,1.05),(axes[1,1],'outer_dev_negative_load',2.4,2.45)]:
            _,b,_=traces[(seed,case)];mask=(b['time_s']>=start)&(b['time_s']<=end)
            ax.plot(b['time_s'][mask],b['outer_command_a'][mask],label=f'L1 seed{seed}')
    axes[0,0].plot(pi['time_s'],pi['omega'],'k--',label='matched PI')
    axes[0,0].plot(pi['time_s'],pi['reference_rad_s'],color='gray',ls=':',label='reference')
    for ax,title,y in zip(axes.ravel(),['Negative load: persistent speed offset','Negative load: reconstructed memory','Startup tail: command detail','Negative-load tail: command detail'],['Speed(rad/s)','z (dimensionless)','iq reference(A)','iq reference(A)']):
        ax.set_title(title);ax.set_xlabel('Time(s)');ax.set_ylabel(y);ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Selected absolute-error policies: development-trace diagnosis')
    for ext in ['png','pdf','svg']:fig.savefig(output/f'l1_diagnosis.{ext}',dpi=160)
    plt.close(fig)
    summary=dict(max_abs_memory=max(max(abs(r['z_min']),abs(r['z_max'])) for r in rows),
        max_abs_event_tail_bias_rad_s=max(r['max_abs_event_tail_bias_rad_s'] for r in rows),
        cases=len(rows),memory_saturated_cases=sum(r['z_bound_fraction']>0 for r in rows))
    save(output/'summary.json',summary);print(summary)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('results/bldc/outer-reward-v1'));p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.source,a.output)
