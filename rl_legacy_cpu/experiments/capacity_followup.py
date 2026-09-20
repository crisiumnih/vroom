"""Bounded, one-shot fresh qualification after frozen capacity selection."""
import argparse
import copy
import importlib.metadata
import json
import time
from pathlib import Path

import torch
from stable_baselines3 import DDPG
from benchmarks.bldc.run import ROOT, provenance, sha, write_trace, write_csv
from benchmarks.bldc.cascaded import CascadedPIController
from benchmarks.bldc.current_rl import capacity_study, speed_bridge
from benchmarks.bldc.current_rl.repeat_study import read, save, now, verify_run
from benchmarks.bldc.current_rl.study import compare


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    save(temporary, value)
    temporary.replace(path)


def current_cases():
    return [dict(name=f'fresh_capacity_current_{i}', speed_rad_s=s, duration_s=.26,
        reference=[[0., [0., 0.]], [.04, [.2, 1.1]], [.115, [-.3, -1.05]],
                   [.185, [.1, .65]], [.235, [0., 0.]]])
        for i, s in enumerate([-9.5, -6.5, -1.5, 1.5, 4.5, 9.5])]


def speed_cases():
    return [
        dict(name='capacity_speed_start', duration_s=1.5, reference=[[0.,0.],[.18,-4.5]], disturbance=[[0.,0.]], stress=False),
        dict(name='capacity_speed_steps', duration_s=2.9, reference=[[0.,0.],[.18,5.5],[1.05,1.5],[2.,-2.5]], disturbance=[[0.,0.]], stress=False),
        dict(name='capacity_speed_load', duration_s=2.5, reference=[[0.,0.],[.18,-5.5]], disturbance=[[0.,0.],[1.,-.025],[1.7,0.]], stress=False),
        dict(name='capacity_speed_reverse', duration_s=2.9, reference=[[0.,0.],[.18,4.5],[1.1,-4.5],[2.,1.5]], disturbance=[[0.,0.]], stress=False),
        dict(name='capacity_speed_stress', duration_s=3., reference=[[0.,0.],[.18,-95.],[1.8,-4.5]], disturbance=[[0.,0.]], stress=True)]


def prepare(capacity, output):
    capacity, output = Path(capacity).resolve(), Path(output).resolve()
    parent = capacity_study.verify(capacity)
    if sorted(parent['seeds']) != [8, 9]:
        raise ValueError('Expected learning seeds 8 and 9')
    config = read(capacity/parent['records'][0]['arm']/'config.json')
    plant = copy.deepcopy(config['plant'])
    plant['controller']['current_reference_limit_a'] = config['study']['reference_limit_a']
    plant['scenarios'] = speed_cases()
    spec = provenance(dict(plant=plant, study=config['study']))
    spec['packages'].update({n: importlib.metadata.version(n) for n in ('torch', 'stable-baselines3')})
    spec['source_sha256'][str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    spec.update(protocol='capacity-fresh-qualification-v1', capacity=str(capacity),
        capacity_protocol_sha256=sha(capacity/'protocol.json'), seeds=[8, 9],
        current_cases=current_cases(), speed_cases=speed_cases(), plant=plant,
        contract=config['study']['contract'], poll_seconds=15, max_wait_seconds=21600,
        qualification='Every current and hybrid speed case must pass for both seeds of the validation-selected architecture',
        interpretation='New deterministic nominal simulation cases; not plant-uncertainty or measured-motor validation',
        final_test_policy='No retry, reselection, tuning or further learning from final outcomes')
    output.mkdir(parents=True, exist_ok=False)
    save(output/'protocol.json', spec)
    (output/'protocol.sha256').write_text(sha(output/'protocol.json')+'\n')
    atomic(output/'downstream_status.json', dict(status='prepared'))
    return spec


def verify(output):
    output = Path(output); spec = read(output/'protocol.json')
    if sha(output/'protocol.json') != (output/'protocol.sha256').read_text().strip():
        raise ValueError('Changed qualification protocol')
    if sha(Path(spec['capacity'])/'protocol.json') != spec['capacity_protocol_sha256']:
        raise ValueError('Changed parent protocol')
    for rel, digest in spec['source_sha256'].items():
        if sha(ROOT/rel) != digest:
            raise ValueError(f'Changed evaluation source: {rel}')
    for name, version in spec['packages'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f'Changed package: {name}')
    capacity_study.verify(Path(spec['capacity']))
    return spec


def readiness(capacity):
    capacity = Path(capacity)
    if (capacity/'manifest.json').exists() and read(capacity/'manifest.json')['status']=='failed':
        raise RuntimeError('Parent queue failed; no qualification')
    if not (capacity/'queue_status.json').exists():
        return False
    queue = read(capacity/'queue_status.json')
    if any(r.get('status')=='failed' or r.get('returncode') != 0 for r in queue['results']):
        raise RuntimeError('Parent learning job failed; no qualification')
    return (queue['completed_jobs']==queue['total_jobs'] and
            (capacity/'selection.json').exists() and (capacity/'selection.sha256').exists())


def selected_rows(spec):
    capacity = Path(spec['capacity']); parent = capacity_study.verify(capacity)
    if not readiness(capacity) or read(capacity/'manifest.json')['status']!='completed':
        raise RuntimeError('Parent queue incomplete')
    if sha(capacity/'selection.json') != (capacity/'selection.sha256').read_text().strip():
        raise ValueError('Changed development selection')
    selection = read(capacity/'selection.json')
    expected = {(r['label'],r['seed']) for r in parent['records']}
    rows = selection['rows']
    if len(rows)!=len(expected) or {(r['label'],r['seed']) for r in rows} != expected:
        raise ValueError('Missing or duplicate selected jobs')
    queue = read(capacity/'run_summary.json')
    if len(queue)!=len(expected) or {r['arm'] for r in queue}!={r['arm'] for r in parent['records']} or any(r['returncode']!=0 for r in queue):
        raise RuntimeError('Incomplete parent completion records')
    for row in rows:
        arm = capacity/f"{row['label']}-seed{row['seed']}"
        if Path(row['parent']).resolve()!=arm.resolve():
            raise ValueError('Unexpected selected model path')
        manifest = verify_run(arm)
        if manifest['selected'] != row['selected']:
            raise ValueError('Selected checkpoint identity mismatch')
        metrics = read(arm/'direct/validation'/str(row['selected']['step'])/'metrics.json')
        if row['passed_cases']!=sum(c['passed'] for c in metrics) or row['case_count']!=len(metrics):
            raise ValueError('Changed selected case counts')
        config = read(arm/'config.json')
        if config['study']['contract']!=spec['contract']:
            raise ValueError('Changed encoder contract')
        plant=copy.deepcopy(config['plant'])
        plant['controller']['current_reference_limit_a']=config['study']['reference_limit_a']
        plant['scenarios']=spec['speed_cases']
        if plant!=spec['plant']:
            raise ValueError('Changed plant')
    scores=capacity_study.aggregate_scores(rows,parent['widths'],parent['seeds'])
    winner=min(scores,key=scores.get)
    if scores!=selection['scores'] or winner!=selection['winner']:
        raise ValueError('Changed capacity ranking')
    if scores[winner][0]!=0:
        raise RuntimeError('Selected capacity does not pass development at both seeds')
    return selection, [r for r in rows if r['label']==winner]


def qualified(rows, spec):
    if sorted(r['seed'] for r in rows)!=sorted(spec['seeds']):
        return False
    for row in rows:
        for kind in ('current','speed'):
            expected=[c['name'] for c in spec[kind+'_cases']]
            actual=[c['case'] for c in row[kind]]
            if sorted(actual)!=sorted(expected):
                return False
            if not all(c['passed'] and c['pi']['completed'] and c['rl']['completed'] for c in row[kind]):
                return False
    return True


def run(output):
    output=Path(output); spec=verify(output); selection, selected=selected_rows(spec)
    with (output/'evaluation_started.json').open('x') as stream:
        json.dump(dict(started_utc=now()),stream)
    dest=output/'evaluation';dest.mkdir(exist_ok=False)
    save(dest/'selection_before_test.json',dict(selection=selection,sha256=sha(Path(spec['capacity'])/'selection.json')))
    atomic(dest/'status.json',dict(status='running',started_utc=now()))
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rows=[]
    try:
        for row in selected:
            arm=Path(row['parent']);cfg=read(arm/'config.json')
            model=DDPG.load(arm/'direct/best_model.zip',device='cpu')
            out=dest/f"{row['label']}_seed{row['seed']}";out.mkdir()
            current=compare(model,cfg['plant'],cfg['study'],'direct',spec['current_cases'],out/'current')
            speed=[]
            for case in spec['speed_cases']:
                pair={}
                for branch, controller in [('pi',CascadedPIController(spec['plant'],spec['plant']['controller'])),
                    ('rl',speed_bridge.HybridController(spec['plant'],cfg['study'],model))]:
                    trace,metrics=speed_bridge.rollout(spec['plant'],case,controller,cfg['study']['phase_limit_a'])
                    write_trace(out/f"{case['name']}_{branch}.csv.gz",trace);pair[branch]=metrics
                violations,comparable=speed_bridge.gates(pair['rl'],pair['pi'],cfg['study']['phase_limit_a'])
                speed.append(dict(case=case['name'],**pair,violations=violations,comparable=comparable,
                                  passed=comparable and not violations,stress=case['stress']))
                atomic(out/'speed_metrics.json',speed)
            rows.append(dict(label=row['label'],seed=row['seed'],current=current,speed=speed))
            atomic(dest/'metrics.json',rows)
        result=dict(status='completed',completed_utc=now(),winner=selection['winner'],qualified=qualified(rows,spec))
        table=[]
        for row in rows:
            for kind in ('current','speed'):
                key='rmse_a' if kind=='current' else 'rmse_rad_s'
                for c in row[kind]:
                    table.append(dict(label=row['label'],seed=row['seed'],kind=kind,case=c['case'],passed=c['passed'],
                        violations=';'.join(c['violations']),pi_rmse=c['pi'].get(key),rl_rmse=c['rl'].get(key),units='A' if kind=='current' else 'rad/s'))
        write_csv(dest/'cases.csv',table)
        lines=['# Fresh capacity-winner qualification','',f"Architecture: **{result['winner']}**. Qualification: **{'PASS' if result['qualified'] else 'FAIL'}**.",'',
               '| Seed | Current cases passed | Hybrid speed cases passed |','|---|---:|---:|']
        for row in rows:
            lines.append(f"| {row['seed']} | {sum(c['passed'] for c in row['current'])}/6 | {sum(c['passed'] for c in row['speed'])}/5 |")
        lines+=['','Both seeds must pass every case. These are deterministic tests of an assumed nominal plant.',
                'Outer speed PI remains. No outer RL, measured-motor validity, parameter robustness or post-test reselection is claimed.',
                'The -95 rad/s stress command is unreachable; its gates assess relative behavior and recovery.']
        (output/'README.md').write_text('\n'.join(lines)+'\n')
        atomic(dest/'status.json',result)
        save(dest/'artifacts.json',{str(f.relative_to(dest)):sha(f) for f in dest.rglob('*') if f.is_file()})
        return result
    except BaseException as exc:
        atomic(dest/'status.json',dict(status='failed',failed_utc=now(),failure=f'{type(exc).__name__}: {exc}'))
        raise


def finish(output, timeout_s=21600):
    output=Path(output);spec=verify(output)
    if not 0<=timeout_s<=spec['max_wait_seconds']:
        raise ValueError('Wait outside frozen bound')
    started=dict(started_utc=now(),timeout_s=timeout_s,protocol_sha256=sha(output/'protocol.json'))
    with (output/'downstream_started.json').open('x') as stream:
        json.dump(started,stream,indent=2)
    atomic(output/'downstream_status.json',dict(status='waiting_for_training_selection',**started))
    begin=time.monotonic()
    try:
        while not readiness(Path(spec['capacity'])):
            if time.monotonic()-begin>=timeout_s:
                raise TimeoutError('Bounded wait expired; no retry or training restart')
            time.sleep(min(spec['poll_seconds'],max(0,timeout_s-(time.monotonic()-begin))))
        atomic(output/'downstream_status.json',dict(status='evaluating_frozen_final_cases',**started))
        result=run(output)
        atomic(output/'downstream_status.json',dict(status='completed',completed_utc=now(),qualification=result,**started))
    except BaseException as exc:
        atomic(output/'downstream_status.json',dict(status='failed',failed_utc=now(),failure=f'{type(exc).__name__}: {exc}',**started))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run','finish'])
    parser.add_argument('--capacity',type=Path,default=Path('results/bldc/current-capacity-v1'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--timeout-s',type=float,default=21600)
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.capacity,args.output)
    elif args.command=='run':run(args.output)
    else:finish(args.output,args.timeout_s)


if __name__=='__main__':main()
