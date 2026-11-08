"""Separate historical complete5, causal dispatch and pure-control landing evidence."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();read=lambda p:json.loads(Path(p).read_text())
    job=read(a.run/'job.json')
    if job['status']!='SUCCEEDED' or job['exit_code']!=0:raise ValueError('Full native completion required')
    result=read(a.run/'pure_control_result.json');meta=read(a.run/'pure_control_ground_forces.json')
    if not meta['complete'] or not meta['interval_aggregates_complete']:raise ValueError('Incomplete physical contact observation')
    commands=np.load(a.run/'pure_control_commands.npz');dispatch=np.load(a.run/'sequence_dispatch.npz')
    forces=np.load(a.run/'pure_control_ground_forces.npz');metrics=read(a.run/'metrics.json')
    rows=[json.loads(line) for line in (a.run/'pure_control_diagnostics.jsonl').read_text().splitlines()]
    limits=np.asarray(read(a.run/'native_command_inputs.json')['joint_limits_rad'])
    decoded=limits[:,0]+.5*(dispatch['final']+1)*np.diff(limits).ravel()
    errors=[];ownership=np.zeros(dispatch['first_episode'].shape,bool)
    for i,(tick,lane) in enumerate(zip(commands['tick'],commands['lane'])):
        ownership[tick,lane]=True;errors.append(float(np.max(abs(decoded[tick,lane]-commands['final'][i]))))
    unchanged=np.array_equal(dispatch['student'][~ownership],dispatch['final'][~ownership])
    if not unchanged or max(errors,default=0)>2e-6:raise ValueError('Dispatch did not match declared ownership')
    head=[meta['body_names'].index(x) for x in ['Head_lower','Head_upper']]
    nonfoot=[i for i,n in enumerate(meta['body_names']) if not n.endswith('_foot')]
    other=[i for i in nonfoot if i not in head]
    cases=[]
    for lane in range(128):
        own=np.flatnonzero(ownership[:,lane]);start=int(own[0]) if len(own) else None
        events=[r for r in rows if r['lane']==lane]
        if start is not None and not ownership[start:,lane][dispatch['first_episode'][start:,lane]].all():
            raise ValueError('Recovery gave commands back before episode ended')
        end=len(ownership) if start is None else int(np.flatnonzero(dispatch['first_episode'][:,lane])[-1])+1
        def stats(indices,begin,stop):
            f=forces['interval_peak_normal_n'][begin:stop,lane][:,indices]
            return dict(peak_n=float(f.max()) if f.size else 0.,
                any_nonzero=bool((f>0).any()),any_over1n=bool((f>1).any()),
                physics_steps_over1n=int(forces['interval_frames_over1n'][begin:stop,lane][:,indices].sum()))
        post_head=stats(head,start or 0,end) if start is not None else None
        post_other=stats(other,start or 0,end) if start is not None else None
        stable=max([r['phase']['control_sample_stable_duration_s'] for r in events],default=0.)
        final_stable=events[-1]['phase']['control_sample_stable_duration_s'] if events else 0.
        cases.append(dict(case=lane,takeover_control_tick=start,head_before_takeover=stats(head,0,start or end),
            head_after_takeover=post_head,other_nonfoot_after_takeover=post_other,
            longest_control_sample_stable_s=stable,final_control_sample_stable_s=final_stable,
            no_recorded_nonfoot_ground_force=(not post_head['any_nonzero'] and not post_other['any_nonzero']) if start is not None else None,
            sampled_stability_is_not_formal_acceptance=True))
    output=dict(passed_dispatch_integrity=True,complete5=metrics['complete5'],execution=result,cases=cases,
        unchanged_commands_outside_ownership=unchanged,max_command_error_rad=max(errors,default=0),
        full_ground_initialized_episodes=True,training=False,valid_expert_labels=False,
        strict_landing_acceptance_not_automatically_inferred=True)
    temp=a.output.with_suffix('.tmp');temp.write_text(json.dumps(output,indent=2)+'\n');temp.replace(a.output)
    print(json.dumps(dict(dispatch_passed=True,complete5=metrics['complete5'],owned_cases=len(result['owned_cases']),
        no_nonfoot_cases=sum(c['no_recorded_nonfoot_ground_force'] is True for c in cases))))


if __name__=='__main__':main()
