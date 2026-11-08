"""Read actual release states and distinguish ground, net and support failures.

No environment reset/state teleport or outcome relabeling. Court geometry on
recorded states is diagnostic; it is not a replacement for native net contacts.
The release-state bank contains failures, not expert demonstrations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context,state_from_row
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.authored_net_clearance_v1 import AuthoredNetClearanceV1
from scripts.event_recovery_manager_v1 import authored_clearance_state
from scripts.audit_joint_sequence_proposal_v1 import plain
from quadsmash.physics.court_v4 import load_court


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['run','model-inputs','native-audit','baseline','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();read=lambda p:json.loads(Path(p).read_text())
    if a.output.exists():raise ValueError('Fresh analysis output required')
    model_inputs=read(a.model_inputs)
    s=read(read(model_inputs['configuration']['model_plan'])['source'])
    *_,context=source_context(s['configuration']['source']);initialize(*context)
    geometry=_WORKER['geometry'];net=AuthoredNetClearanceV1(geometry,load_court())
    def arrays(path):
        with np.load(path) as z:return {k:z[k].copy() for k in z.files}
    data=arrays(a.run/'pure_control_commands.npz')
    force=arrays(a.run/'pure_control_ground_forces.npz')
    meta=read(a.run/'pure_control_ground_forces.json');audit=read(a.native_audit)
    episodes={r['env_id']:r for r in read(a.run/'episodes.json')}
    baseline={r['env_id']:r for r in read(a.baseline/'episodes.json')}
    rows=[json.loads(line) for line in (a.run/'pure_control_diagnostics.jsonl').read_text().splitlines()]
    if len(rows)!=len(data['tick']):raise ValueError('State/diagnostic alignment required')
    for i,row in enumerate(rows):
        if (row['tick'],row['lane'])!=(data['tick'][i],data['lane'][i]):raise ValueError('Misaligned diagnostic')
    head=[meta['body_names'].index(n) for n in ['Head_lower','Head_upper']]
    other=[i for i,n in enumerate(meta['body_names']) if not n.endswith('_foot') and i not in head]
    cases=[];queries=[]
    for lane in sorted(set(data['lane'].tolist())):
        ids=np.flatnonzero(data['lane']==lane);first=int(ids[0]);tick=int(data['tick'][first]);episode=episodes[lane]
        def query(i):
            return dict(q=data['q'][i],v=data['v'][i],target=data['previous'][i],contacts=rows[i]['phase']['ground_contacts'])
        initial=query(first);state=state_from_row(initial)
        surfaces=authored_clearance_state(geometry,state)
        near=net.evaluate(state.q,near_distance_m=.08)
        initial_net=min(near.items(),key=lambda x:x[1]['distance_m'])
        first_net=None
        # Recorded states precede the respective command. Geometry evidence
        # is sampled at 100 Hz and cannot time a native net force precisely.
        for i in ids:
            checks=net.evaluate(data['q'][i],near_distance_m=0.)
            penetrating=[dict(body=b,**r) for b,r in checks.items() if r['exact'] and r['distance_m'] < -1e-6]
            if penetrating:
                first_net=dict(tick=int(data['tick'][i]),time_s=float(data['time_s'][i]),shapes=penetrating)
                break
        end=int(data['tick'][ids[-1]])+1
        def first_ground(indices,start,stop):
            x=force['interval_first_frame_over1n'][start:stop,lane][:,indices]
            hit=x[x>=0]
            if not len(hit):return None
            frame=int(hit.min());at=np.argwhere(x==frame)[0]
            body=meta['body_names'][indices[int(at[1])]]
            return dict(physics_frame=frame,time_s=frame*meta['physics_rate_aggregation_s'],body=body,
                elapsed_from_takeover_s=frame*meta['physics_rate_aggregation_s']-tick*.01)
        comparisons={}
        for key in ['had_contact','string_hit','airborne_string_hit','contact_time_s','release_time_s','outgoing_velocity_mps']:
            old=baseline[lane][key];new=episode[key]
            comparisons[key]=bool(np.allclose(np.asarray(old),np.asarray(new),rtol=0,atol=1e-5,equal_nan=True))
        record=dict(case=lane,takeover_tick=tick,takeover_time_s=float(data['time_s'][first]),
            release_geometry=surfaces,release_nearest_net=dict(body=initial_net[0],**initial_net[1]),
            native_head_before=first_ground(head,0,tick),native_head_after=first_ground(head,tick,end),
            native_other_after=first_ground(other,tick,end),
            native_robot_net_fault=episode['robot_net_fault'],first_sampled_net_penetration=first_net,
            net_geometry_is_not_native_force_time=True,
            solver_accepted_steps=sum(rows[i]['diagnostic']['solver_accepted'] for i in ids),
            fallback_steps=sum(not rows[i]['diagnostic']['solver_accepted'] for i in ids),
            causal_ball_prefix_comparison_to_baseline=comparisons,
            longest_control_sample_stable_s=audit['cases'][lane]['longest_control_sample_stable_s'],
            no_recorded_nonfoot_ground_force=audit['cases'][lane]['no_recorded_nonfoot_ground_force'])
        cases.append(record)
        queries.append(dict(case=lane,source_row=first,control_tick=tick,time_s=float(data['time_s'][first]),
            query=initial,measured_ground_force_world=data['ground_force'][first],
            controller_phase=rows[first]['phase'],native_state=True,native_teleportation=False,
            valid_expert_label=False))
    summary=dict(owned_cases=len(cases),native_robot_net_faults=sum(r['native_robot_net_fault'] for r in cases),
        sampled_net_penetration_cases=sum(r['first_sampled_net_penetration'] is not None for r in cases),
        native_head_after_cases=sum(r['native_head_after'] is not None for r in cases),
        native_other_after_cases=sum(r['native_other_after'] is not None for r in cases),
        clean_and_stable_sampled_cases=[r['case'] for r in cases if r['no_recorded_nonfoot_ground_force'] and r['longest_control_sample_stable_s']>=.8],
        ball_prefix_comparison_pass_cases=sum(all(r['causal_ball_prefix_comparison_to_baseline'].values()) for r in cases),
        ball_prefix_comparison_mismatch_cases=[r['case'] for r in cases if not all(r['causal_ball_prefix_comparison_to_baseline'].values())])
    paths=[a.model_inputs,a.native_audit,a.run/'pure_control_commands.npz',a.run/'pure_control_diagnostics.jsonl',
        a.run/'pure_control_ground_forces.npz',a.run/'pure_control_ground_forces.json',a.run/'episodes.json',a.baseline/'episodes.json',Path(__file__),ROOT/'scripts/authored_net_clearance_v1.py']
    pins={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}
    a.output.mkdir(parents=True)
    for name,payload in [('failure_chain.json',dict(summary=summary,cases=cases,source_sha256=pins)),
        ('release_state_bank.json',dict(queries=queries,source_sha256=pins,valid_expert_labels=False,
            scope='Observed native post-separation states for model planning; no native teleport'))]:
        path=a.output/name;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(plain(payload),indent=2,allow_nan=False)+'\n');tmp.replace(path)
    print(json.dumps(summary))


if __name__=='__main__':main()
