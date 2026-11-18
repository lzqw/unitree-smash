"""Read-only exact-prefix substep response audit; no fits or policy updates.

A sensor's summed force direction is NOT a contact normal, and its averaged
contact point is NOT an individual PhysX manifold point. Keep this distinction.
"""
import argparse,hashlib,json
from pathlib import Path
import torch
from quadsmash.rl.supermimic_native_response_data_v1 import load_response_data


def first_index(mask):
    found=mask.nonzero().flatten()
    return int(found[0]) if len(found) else None


def validate_trace(run):
    trace=run['raw']['substep_trace'];x=trace['tensors']
    if trace['schema']!='native-one-control-tick-substeps-v1':raise ValueError('Unknown trace schema')
    if trace['count']!=161 or trace['physics_dt_s']!=.0000625:raise ValueError('Changed original control/physics window')
    frame=x['physics_frame']
    if frame.shape!=(161,) or not torch.equal(frame,torch.arange(trace['start_frame'],trace['start_frame']+161)):
        raise ValueError('Substep frame gap, duplicate or wrong origin')
    for key,v in x.items():
        if key=='physics_frame':continue
        if v.shape[:2]!=(161,128):raise ValueError('Substep lane/clock shape '+key)
        if key!='racket_contact_position_w' and not torch.isfinite(v).all():raise ValueError('Nonfinite observed state '+key)
    if not torch.equal(torch.isfinite(x['racket_contact_position_w']).all(-1),x['racket_contact_position_valid']):
        raise ValueError('Contact-position validity differs from recorded data')
    before=run['raw']['start']['before']
    # Trace starts at the exact captured state, before the perturbed command.
    for key,truth in [('shuttle_state_w',before['shuttle_state_w']),('ball_clock',before['auxiliary']['environment']['ball_clock'])]:
        if not torch.equal(x[key][0],truth):raise ValueError('Trace start not aligned: '+key)
    if not torch.equal(x['shuttle_state_w'][-1,:,0:3],run['raw']['next_state'][:,49:52]):raise ValueError('Trace end position differs from transition')
    if not torch.equal(x['shuttle_state_w'][-1,:,7:10],run['raw']['next_state'][:,52:55]):raise ValueError('Trace end velocity differs from transition')
    return x


def analyze(root,split_path,output):
    if output.exists():raise FileExistsError(output)
    _,provenance=load_response_data(root,split_path)
    runs={b:torch.load(root/b/'native_probe.pt',map_location='cpu',weights_only=True) for b in ('nominal_a','nominal_b','minus','plus')}
    traces={b:validate_trace(r) for b,r in runs.items()}
    repeat={}
    for key,a in traces['nominal_a'].items():
        b=traces['nominal_b'][key]
        same=(a==b)|(torch.isnan(a)&torch.isnan(b)) if a.is_floating_point() else a==b
        repeat[key]=bool(same.all())
    if not all(repeat.values()):raise ValueError('Nominal substep replay did not repeat exactly')
    physics=json.loads(Path('configs/shuttle_physics_v2.json').read_text())
    dt=float(physics['native_physics_dt_s']);threshold=float(physics['contact_force_threshold_n'])
    if dt!=runs['nominal_a']['raw']['substep_trace']['physics_dt_s']:raise ValueError('Changed native clock contract')
    command=runs['plus']['raw']['final_joint_targets']-runs['minus']['raw']['final_joint_targets']
    delta=traces['plus']['shuttle_state_w'][:,:,7:10]-traces['minus']['shuttle_state_w'][:,:,7:10]
    qdelta=traces['plus']['joint_position']-traces['minus']['joint_position']
    lanes=[]
    for lane in range(128):
        branch={}
        for name,trace in traces.items():
            force=trace['racket_force_w'][:,lane];touch=force.norm(dim=-1)>threshold
            edges=touch[1:]&~touch[:-1]
            first=first_index(edges)
            if first is not None:first+=1
            active=touch.nonzero().flatten()
            row=dict(initial_contact=bool(touch[0]),new_contact_edges=(edges.nonzero().flatten()+1).tolist(),
                first_new_contact_index=first,contact_frame_count=int(touch.sum()),
                max_force_n=float(force.norm(dim=-1).max()),
                initial_ball_spin=trace['shuttle_state_w'][0,lane,10:13].tolist(),
                native_frame_hit=bool(runs[name]['raw']['task_after']['frame_hit'][lane]),
                native_string_hit=bool(runs[name]['raw']['task_after']['string_hit'][lane]))
            if first is not None:
                row.update(first_ball_clock=float(trace['ball_clock'][first,lane]),
                    first_frame_delta_s=first*dt,
                    event_neighborhood=[dict(index=i,ball_velocity=trace['shuttle_state_w'][i,lane,7:10].tolist(),
                        ball_spin=trace['shuttle_state_w'][i,lane,10:13].tolist(),force_w=force[i].tolist(),
                        averaged_contact_position_w=trace['racket_contact_position_w'][i,lane].tolist() if bool(trace['racket_contact_position_valid'][i,lane]) else None,
                        racket_linear_velocity=trace['racket_linear_velocity_w'][i,lane].tolist(),
                        racket_angular_velocity=trace['racket_angular_velocity_w'][i,lane].tolist())
                        for i in range(max(0,first-2),min(161,int(active[-1])+3))])
            branch[name]=row
        n=branch['nominal_a']['first_new_contact_index']
        rows=sorted(set([0,160]+([] if n is None else [max(0,n-1),n,min(160,n+1),min(160,n+4)])))
        lanes.append(dict(lane=lane,split='validation' if lane in provenance['validation_lanes'] else 'training',
            final_target_pair_delta_rad=float(command[lane].norm()),
            first_nonzero_joint_response_index=first_index(qdelta[:,lane].norm(dim=-1)>0),
            first_nonzero_ball_response_index=first_index(delta[:,lane].norm(dim=-1)>0),
            final_ball_response=delta[-1,lane].tolist(),branch=branch,
            response_neighborhood=[dict(index=i,ball_delta=delta[i,lane].tolist(),joint_position_delta_norm=float(qdelta[i,lane].norm())) for i in rows]))
    result=dict(scope=__doc__,source=str(root),provenance=provenance,substep_repeat_exact=repeat,
        contact_force_threshold_n=threshold,threshold_source='unchanged configs/shuttle_physics_v2.json',
        physics_dt_s=dt,actor_updates=0,model_updates=0,lanes=lanes,
        evidence_scope='Paired native data; no model passing receipt or task score')
    tmp=output.with_suffix('.pending');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(output)
    print(json.dumps(dict(substep_repeat_exact=all(repeat.values()),contact_lanes=[r['lane'] for r in lanes if r['branch']['nominal_a']['first_new_contact_index'] is not None],output=str(output))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--split-manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();analyze(args.run,args.split_manifest,args.output)
