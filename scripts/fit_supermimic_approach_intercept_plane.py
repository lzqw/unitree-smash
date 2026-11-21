"""Fit an incoming target-plane offset on training launches, then measure native contact timing."""
import argparse
import json
import statistics
from pathlib import Path

import torch

from quadsmash.rl.supermimic_action_identification_dataset_v1 import _load_branch
from quadsmash.rl.supermimic_physical_imagination_state_v1 import PhysicalImaginationStateV1
from quadsmash.rl.supermimic_toi_racket_objective import flight_intercept
from scripts.audit_supermimic_landing_command_response import load_world
from scripts.declare_supermimic_contact_training_cases import resolve_probe_cases
from scripts.audit_supermimic_native_window_response import complete_window
from scripts.evaluate_supermimic_fresh_wm_panel import digest


def measure(root, template, launch, task, held, training, offset):
    read=lambda path:json.loads(path.read_text())
    status=read(root/'status.json')
    if status['status']!='COLLECTION_FINISHED':
        raise ValueError('Expected complete native probes')
    nominal,=[r for r in status['completed'] if r['branch']=='nominal_a']
    directory=root/nominal['spec']
    if nominal['exit_code'] or digest(directory/'nominal_a/native_probe.pt')!=nominal['probe_sha256']:
        raise ValueError('Native nominal has changed')
    run=_load_branch(directory,'nominal_a')
    spec, raw=run['spec'],run['raw']
    ids=set(spec['lanes'])
    seed = run['seed']
    expected = task
    if training:
        _, panel_hash = resolve_probe_cases(spec, [])
        if not panel_hash or ids & held or spec['data_role'] != 'contact_model_training':
            raise ValueError('Expected an identified training-only native contact panel')
        expected = dict(task, contact_training_only=True, contact_training_panel_sha256=panel_hash)
        seed = spec['contact_training_panel']['seed']
    elif not ids <= held or spec['data_role'] != 'development_action_response':
        raise ValueError('Expected excluded development lanes')
    if run['task_contract'] != expected:
        raise ValueError('Native physics or task contract changed')
    state=PhysicalImaginationStateV1.from_record(raw['start'],executor_template=template,launch=launch,device='cpu')
    pred=flight_intercept(state,.64,plane_offset_m=offset)
    steps=raw['substep_trace']['count']-1
    traces=complete_window(raw,spec,steps)
    cases=[]
    elapsed=state.clock_goal_context['elapsed'].reshape(-1)
    for i,lane in enumerate(spec['lanes']):
        label=raw['task_after']
        if raw['start']['before']['task']['had_contact'][lane] or not label['had_contact'][lane]:
            continue
        stamp=label['contact_time'][lane]
        hits=[(j,int(k)+1) for j,tr in enumerate(traces) for k in torch.where(tr['tensors']['ball_clock'][1:,i]==stamp)[0]]
        if len(hits)!=1:
            raise ValueError('Contact latch differs from recorded native clock')
        j,k=hits[0]
        native_position=traces[j]['tensors']['shuttle_state_w'][k,i,:3]
        # Compare on elapsed control time; ball_clock accumulates at the higher physics rate.
        native_time=float(elapsed[lane])+j*task['control_dt_s']+k*traces[j]['physics_dt_s']
        valid=bool(pred['valid'][lane])
        cases.append(dict(lane=lane,qualified=bool(label['airborne_string_hit'][lane] and label['legal_initial_contact'][lane]
                                                    and not label['frame_hit'][lane]),valid=valid, tick=j, substep=k,
                          error_ms=1000*(float(pred['time'][lane])-native_time) if valid else None,
                          native_minus_plane_m=float(native_position[0]-pred['position'][lane,0]) if valid else None))
    summary={}
    for key,items in [('any_first_contact',cases),('airborne_string',[r for r in cases if r['qualified']])]:
        values=[r['error_ms'] for r in items if r['valid']]
        summary[key]=dict(labeled=len(items),forecast=len(values),median_error_ms=statistics.median(values) if values else None,
            median_absolute_error_ms=statistics.median(map(abs,values)) if values else None,
            within_one_control_step=sum(abs(v)<=task['control_dt_s']*1000 for v in values))
    return dict(nominal_sha256=nominal['probe_sha256'],seed=seed,lanes=spec['lanes'],status_sha256=digest(root/'status.json'),
                metrics=summary,cases=cases)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    for key in ('run','checkpoint','output'):
        cli.add_argument('--'+key,type=Path,required=True)
    for key in ('training','development'):
        cli.add_argument('--'+key,type=Path,nargs='+',required=True)
    args=cli.parse_args()
    torch.set_num_threads(2)
    ck,_,_,template,launch,contract=load_world(args.run,args.checkpoint)
    def panel(roots,training,offset):
        return [measure(root,template,launch,contract['task_contract'],set(ck['config']['wm_holdout_lanes']),training,offset) for root in roots]
    train=panel(args.training,True,0.)
    if len({t['seed'] for t in train})!=len(train):
        raise ValueError('Repeated training launch seed')
    offsets=[r['native_minus_plane_m'] for t in train for r in t['cases'] if r['qualified'] and r['valid']]
    correction=statistics.median(offsets)
    dev=panel(args.development,False,0.)
    if {t['seed'] for t in train} & {t['seed'] for t in dev}:
        raise ValueError('Development prefix used to fit contact timing')
    out=dict(schema='training-native-contact-target-plane-v1',plane_offset_m=correction,fit_samples=len(offsets),
             checkpoint_sha256=digest(args.checkpoint),training=train,development_baseline=dev,
             development_shifted=panel(args.development,False,correction),actor_updates=0,promotable=False)
    with args.output.open('x') as f:
        json.dump(out,f,indent=2,allow_nan=False)


if __name__=='__main__':
    main()
