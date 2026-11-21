#!/usr/bin/env python3
"""Separate native-link reconstruction from Body WM prediction on held G07 commands."""
import argparse
import json
from pathlib import Path

import torch
from audit_supermimic_native_window_response import BRANCHES, complete_window
from audit_supermimic_paired_body_transfer import response_metrics
from evaluate_supermimic_contact_rollout import BallImpulsePriorV1, PhysicalWorldV2, predicted_racket_path, racket, sha, tick
from quadsmash.rl.supermimic_action_identification_dataset_v1 import _load_branch, _same_contract
from quadsmash.rl.supermimic_contact_pilot_v1 import _quat_rotate, _quat_rotate_inverse
from quadsmash.rl.supermimic_physical_dataset_v1 import encode_physical
from quadsmash.rl.supermimic_racket_material_contract_v1 import validate_contract


def racket_fields(pose):
    center,quat,linear,angular=pose
    axis=torch.zeros_like(center)
    axis[:,0]=1  # The string bed occupies the local YZ plane.
    return dict(center=center,normal=_quat_rotate(quat,axis),linear=linear,angular=angular)


def contact_point_fields(pose, lever_local):
    center,q,linear,angular=pose
    arm=_quat_rotate(q[None],lever_local[None])[0]
    normal=_quat_rotate(q[None],q.new_tensor([[1.,0.,0.]]))[0]
    return dict(center=center,linear=linear,angular=angular,normal=normal,
                point_position=center+arm,point_velocity=linear+torch.linalg.cross(angular,arm))


def registered_impacts(report, comparison, response_sha, nominal, steps):
    if response_sha not in comparison['scale_reports_sha256']:
        raise ValueError('Native convergence report belongs to different probes')
    selected=set(comparison['groups']['native_convergent_stable']['case_ids'])
    impacts={}
    for column,c in enumerate(report['cases']):
        if c['lane'] not in selected:
            continue
        tick,frame=divmod(c['signatures']['nominal_a'][3][0],steps)
        tr=nominal[tick]
        if c['group']!='racket_stable':
            raise ValueError('Native contact time switches across branches')
        normal=tr['manifold_frames'][frame+1]
        take=(normal['lane']==column)&(normal['filter_index']==0)&(normal['normal_force_n'].abs()>.001)
        if not take.any():
            raise ValueError('Missing racket contact point at the registered substep')
        weights=normal['normal_force_n'][take].abs()
        point=(weights[:,None]*normal['point_w'][take]).sum(0)/weights.sum()
        t=tr['tensors']
        offset=point-t['racket_position_w'][frame,column]
        lever=_quat_rotate_inverse(t['racket_quaternion_w'][frame,column][None],offset[None])[0]
        impacts[column]=(tick,frame,lever)
    if len(impacts)!=len(selected):
        raise ValueError('Registered lanes absent from native packet')
    return impacts


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('collection','response','body','material','calibration','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--convergence',type=Path,help='Report signed response at the first loaded native substep on the converged three-scale cohort')
    p.add_argument('--attribute-body',action='store_true',
                   help='Also substitute only native root motion or only native articulation on the predicted robot trajectories')
    a=p.parse_args()
    torch.set_num_threads(2)
    read=lambda path:json.loads(path.read_text())
    base=read(a.response)
    status=a.collection/'status.json'
    for field,path in [('status',status),('body',a.body),('material',a.material),('calibration',a.calibration)]:
        if sha(path)!=base[field+'_sha256']:raise ValueError('Changed '+field)
    meta=read(status)
    rows=meta['completed']
    if meta['status']!='COLLECTION_FINISHED' or len(rows)!=4 or {x['branch'] for x in rows}!=set(BRANCHES):
        raise ValueError('Incomplete original collection')
    runs={}
    for row in rows:
        branch=row['branch']
        folder=a.collection/row['spec']
        if sha(folder/branch/'native_probe.pt')!=row['probe_sha256']:raise ValueError('Changed native probe')
        runs[branch]=_load_branch(folder,branch)
    _same_contract(runs)
    task=runs['nominal_a']['task_contract']
    checkpoint=torch.load(a.body,map_location='cpu',weights_only=True)
    lanes=runs['nominal_a']['spec']['lanes']
    if (task!=base['task_contract'] or lanes!=[r['lane'] for r in base['cases']] or task!=checkpoint['source_manifest']['task_contract']
            or set(lanes)&set(checkpoint['calibration']['train_lanes'])):
        raise ValueError('Different G07 panel or body fit identity')
    body=PhysicalWorldV2(task['asset']['sha256']).eval()
    body.load_state_dict(checkpoint['model'])
    suite=Path(task['suite']['path'])
    if sha(suite)!=task['suite']['sha256']:raise ValueError('Different launches')
    launch=torch.tensor([c['launch']['position_m']+c['launch']['velocity_mps'] for c in read(suite)['cases']])[lanes]
    material=torch.tensor(validate_contract(read(a.material),task['asset']['sha256'])['effective'])
    prior=BallImpulsePriorV1().eval()
    cfg=read(a.calibration)
    keys=('racket_position_w','racket_quaternion_w','racket_linear_velocity_w','racket_angular_velocity_w')
    names=['measured','native_fk','body']
    if a.attribute_body:names.extend(('native_root','native_joints'))
    recordings={}
    impacts={}
    samples_at_contact={}
    if a.convergence:
        nominal=runs['nominal_a']
        impacts=registered_impacts(base,read(a.convergence),sha(a.response),
                complete_window(nominal['raw'],nominal['spec'],prior.steps),prior.steps)
    for branch,run in runs.items():
        raw=run['raw']
        windows=complete_window(raw,run['spec'],prior.steps)
        initial=raw['start']['before']['physical']
        predicted=native=encode_physical(initial)[lanes]
        previous,feet,present,elapsed=[initial[k][lanes] for k in ('final_action','foot_contact','ball_present','elapsed_s')]
        first=windows[0]['tensors']['shuttle_state_w'][0]
        balls={name:predicted[:,49:] for name in names}
        spins={name:first[:,10:13] for name in balls}
        samples={name:[] for name in balls}
        impact_rows=samples_at_contact[branch]={name:{} for name in ['native_substep',*names]}
        for ti,(trace,step) in enumerate(zip(windows,raw['step_sequence'],strict=True)):
            targets=step['final_joint_targets'][lanes]
            next_body,contact,_=body(predicted,targets,previous,feet.float(),present,elapsed,launch)
            next_native=step['next_state'][lanes]
            t=trace['tensors']
            endpoints=dict(measured=([t[k][0] for k in keys],[t[k][-1] for k in keys]),
                           native_fk=(racket(native,body.geometry),racket(next_native,body.geometry)),
                           body=(racket(predicted,body.geometry),racket(next_body,body.geometry)))
            if a.attribute_body:
                for name,use_root in [('native_root',True),('native_joints',False)]:
                    poses=[]
                    for observed,estimated in [(native,predicted),(next_native,next_body)]:
                        # The first 13 fields encode the root; FK uses joint q and qdot in fields 13:49.
                        root_part,joint_part=(observed,estimated) if use_root else (estimated,observed)
                        mixed=torch.cat((root_part[:,:13],joint_part[:,13:]),dim=-1)
                        poses.append(racket(mixed,body.geometry))
                    endpoints[name]=poses
            for col,(control,frame,lever) in impacts.items():
                if ti!=control:
                    continue
                impact_rows['native_substep'][col]=contact_point_fields(tuple(t[k][frame,col] for k in keys),lever)
                for tag,(start,end) in endpoints.items():
                    path=predicted_racket_path(prior,tuple(v[col:col+1] for v in start),tuple(v[col:col+1] for v in end))
                    impact_rows[tag][col]=contact_point_fields(tuple(v[frame,0] for v in path),lever)
            for name,(start,end) in endpoints.items():
                ball,_,spin=tick(prior,balls[name],start,end,cfg['fitted'],cfg['solver']['pair_contact_offset_m'],
                    racket_material=material,initial_spin=spins[name],compound_racket=True,return_spin=True)
                balls[name],spins[name]=ball,spin
                samples[name].append(dict(**racket_fields(end),ball=ball[:,3:]))
            native=next_native
            predicted=torch.cat((next_body[:,:49],balls['body']),-1)
            feet=contact.sigmoid()
            previous=step['final_action'][lanes]
            elapsed=elapsed+prior.dt
        recordings[branch]={n:{k:torch.stack([s[k] for s in seq]) for k in seq[0]} for n,seq in samples.items()}
    final=str(base['ticks'][-1])
    cases=base['cases']
    actual=torch.tensor([c['ticks'][final]['native'] for c in cases])
    noise=torch.tensor([c['ticks'][final]['native_repeat'] for c in cases])
    summary={}
    for name in names:
        p1,m1=[recordings[b][name] for b in ('plus','minus')]
        truth_plus,truth_minus=[recordings[b]['measured'] for b in ('plus','minus')]
        ball_delta=(p1['ball']-m1['ball'])[-1]
        if name=='body':
            torch.testing.assert_close(ball_delta,torch.tensor([c['ticks'][final]['body_racket'] for c in cases]),rtol=1e-5,atol=2e-6)
        groups={'all':torch.ones(len(cases),dtype=torch.bool),'racket_stable':torch.tensor([c['group']=='racket_stable' for c in cases])}
        out={}
        for group,mask in groups.items():
            if not mask.any():continue
            stage=dict(outgoing_ball=response_metrics(actual[mask],ball_delta[mask],noise[mask]))
            for k in ('center','normal','linear','angular'):
                ref=(truth_plus[k]-truth_minus[k])[:,mask].flatten(0,1)
                candidate=(p1[k]-m1[k])[:,mask].flatten(0,1)
                repeat=(recordings['nominal_a']['measured'][k]-recordings['nominal_b']['measured'][k])[:,mask].flatten(0,1)
                errors=torch.cat([(recordings[b][name][k][:,mask]-recordings[b]['measured'][k][:,mask]).norm(dim=-1).flatten() for b in BRANCHES])
                stage[k]=dict(native_vector_error_p50=float(errors.median()),native_vector_error_p95=float(errors.quantile(.95)),
                              paired=response_metrics(ref,candidate,repeat))
            out[group]=stage
        summary[name]=out
    at_contact=None
    if a.convergence:
        cols=list(impacts)
        measurements={}
        for tag in ('native_substep',*names):
            measurements[tag]={}
            for field in ('center','normal','linear','angular','point_position','point_velocity'):
                def delta(name, positive, negative):
                    return torch.stack([samples_at_contact[positive][name][c][field]-samples_at_contact[negative][name][c][field] for c in cols])
                if cols:
                    reference=delta('native_substep','plus','minus')
                    pair=delta(tag,'plus','minus')
                    noise=delta('native_substep','nominal_b','nominal_a')
                    measurements[tag][field]=dict(metrics=response_metrics(reference,pair,noise),paired_vectors=pair.tolist())
        at_contact=dict(comparison_sha256=sha(a.convergence),lanes=[lanes[c] for c in cols],
            control_ticks=[base['ticks'][impacts[c][0]] for c in cols],substeps=[impacts[c][1] for c in cols],
            nominal_local_points_m=[impacts[c][2].tolist() for c in cols],measurements=measurements,
            scope='Pre-solve link twist at the first native loaded substep. A fixed nominal material point is shared across all four branches.')
    with a.output.open('x') as f:
        json.dump(dict(groups=summary,first_contact=at_contact,source_sha256=sha(a.response),
                       scope='Identical contact solver; each racket path uses only its two control boundary samples. Pose oracle is diagnostic.',
                       contact_gradients_admitted=False),f,indent=2,allow_nan=False)
        f.write('\n')


if __name__=='__main__':main()
