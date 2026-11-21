"""Inspect native contact geometry, spin and signed accumulated load without redefining stable support."""
import argparse
import json
import math
import statistics
from pathlib import Path

import torch
from scipy.optimize import linear_sum_assignment
from torch.nn.functional import normalize

from audit_supermimic_native_window_response import complete_window, window_signature
from evaluate_supermimic_contact_rollout import sha
from quadsmash.rl.supermimic_action_identification_dataset_v1 import _load_branch, _same_contract
from quadsmash.rl.supermimic_ball_impulse_prior_v1 import BallImpulsePriorV1
from quadsmash.rl.supermimic_ball_geometry_preview_v1 import box_gap, round_gap
from quadsmash.rl.supermimic_contact_pilot_v1 import _quat_rotate_inverse
from quadsmash.rl.supermimic_native_manifold_groups import classify, signatures
from quadsmash.rl.supermimic_native_pair_capture_v1 import BRANCHES


def summarize_trace(tr, tick, mass, rows, geometry):
    t=tr['tensors']
    scale=tr['physics_dt_s']/mass
    normal=t['racket_force_w'][1:];friction=t['racket_friction_force_w'][1:]
    loaded=(normal.norm(dim=-1)>.001)|(friction.norm(dim=-1)>.001)
    for i,lane in loaded.nonzero().tolist():
        nf=tr['manifold_frames'][i+1];ff=tr['racket_friction_frames'][i+1]
        select=(nf['lane']==lane)&(nf['filter_index']==0)&(nf['normal_force_n'].abs()>.001)
        fs=(ff['lane']==lane)&(ff['force_w_n'].norm(dim=1)>.001)
        normals=nf['normal_w'][select]
        points=nf['point_w'][select]
        anchors=ff['point_w_m'][fs]
        center=t['racket_position_w'][i,lane]
        q=t['racket_quaternion_w'][i,lane]
        shuttle=t['shuttle_state_w'][i:i+2,lane]
        rels=[]; tangent=[];local=[]
        projected=[]
        for k in range(2):
            b=shuttle[k];tw=t['racket_angular_velocity_w'][i+k,lane]
            surface=t['racket_linear_velocity_w'][i+k,lane]+torch.linalg.cross(tw.expand_as(anchors),anchors-t['racket_position_w'][i+k,lane])
            rel=b[7:10]+torch.linalg.cross(b[10:13].expand_as(anchors),anchors-b[:3])-surface
            rels.append(rel.tolist())
            if len(points) and len(anchors):
                n=normalize(normals[torch.cdist(anchors,points).argmin(1)],dim=1)
                projected.append(rel-n*(rel*n).sum(1,keepdim=True))
                tangent.append(projected[-1].norm(dim=1).tolist())
            else:
                tangent.append(None)
        if len(anchors):
            local=_quat_rotate_inverse(q.expand(len(anchors),-1),anchors-center).tolist()
        inferred=[]
        if len(points) and len(anchors):
            point_local=_quat_rotate_inverse(q.expand(len(points),-1),points-center)
            gaps=torch.cat((box_gap(point_local,geometry.racket_box_centers,geometry.racket_box_halves),
                round_gap(point_local,geometry.racket_round_centers,geometry.racket_round_axes,
                          geometry.racket_round_halves,geometry.racket_round_radii,geometry.racket_round_capsule)),-1)
            nearest=torch.cdist(anchors,points).argmin(1)
            inferred=gaps.abs().argmin(1)[nearest].tolist()
        along_incident=None
        if projected:
            incident,outgoing=projected
            along=(outgoing*normalize(incident,dim=1)).sum(1)
            along_incident=[float(v) if speed>1e-6 else None for v,speed in zip(along,incident.norm(dim=1),strict=True)]
        bucket=rows[lane]
        impulses=[a[i,lane]*scale for a in (normal,friction)]
        cum=[x+torch.tensor(bucket[-1]['cumulative_dv_mps'][j]) if bucket else x for j,x in enumerate(impulses)]
        bucket.append(dict(frame=tick*len(normal)+i, normal_points=len(points), friction_points=len(anchors),
             signed_post_slip_along_incident_mps=along_incident,separation_m=nf['separation_m'][select].tolist(),
             anchors_local_m=local,inferred_anchor_shape=inferred,normal_w=normals.tolist(),relative_velocity_w_mps=rels,tangent_speed_pre_post_mps=tangent,
             spin_pre_post_rad_s=shuttle[:,10:13].tolist(),normal_dv_mps=impulses[0].tolist(),
             tangent_dv_mps=impulses[1].tolist(),cumulative_dv_mps=[v.tolist() for v in cum]))


def compare(minus,plus):
    result=[]
    negative={x['frame']:x for x in minus}
    positive={x['frame']:x for x in plus}
    for frame in sorted(negative.keys() | positive.keys()):
        if frame not in negative or frame not in positive:
            result.append(dict(frame_minus=frame if frame in negative else None,frame_plus=frame if frame in positive else None))
            continue
        a,b=negative[frame],positive[frame]
        row=dict(frame_minus=a['frame'],frame_plus=b['frame'],friction_counts=[a['friction_points'],b['friction_points']])
        if a['friction_points']==b['friction_points'] and a['friction_points']:
            x=torch.tensor(a['anchors_local_m']); y=torch.tensor(b['anchors_local_m'])
            dist=torch.cdist(x,y)
            i,j=linear_sum_assignment(dist.numpy())
            row['matched_anchor_motion_m']=dist[i,j].tolist()
            if a['inferred_anchor_shape'] and b['inferred_anchor_shape']:
                row['inferred_same_shape']=[a['inferred_anchor_shape'][u]==b['inferred_anchor_shape'][v] for u,v in zip(i,j)]
        if a['normal_points']==b['normal_points'] and a['normal_points']:
            x=normalize(torch.tensor(a['normal_w']),dim=1);y=normalize(torch.tensor(b['normal_w']),dim=1)
            sim=x@y.T
            i,j=linear_sum_assignment(-sim.numpy())
            row['matched_normal_cosine']=sim[i,j].tolist()
        result.append(row)
    return result


def measured_slip_law(cases, friction, sticking):
    """Evaluate the scalar sticking law at the observed contact. Forces are labels for diagnosis."""
    rows = []
    counts = {}
    for case in cases:
        if case['mode'] != 'racket_stable':
            continue
        values = {}
        for branch, events in case['branches'].items():
            if len(events) != 1:
                break
            event = events[0]
            speeds = event['tangent_speed_pre_post_mps'][0]
            if not speeds:
                break
            cap = friction * math.hypot(*event['normal_dv_mps'])
            uncapped = sticking * statistics.mean(speeds)
            values[branch] = dict(predicted=min(uncapped, cap), cap_limited=cap < uncapped,
                                  observed=math.hypot(*event['tangent_dv_mps']),
                                  incident_speed_range_mps=[min(speeds),max(speeds)],
                                  friction_anchors=event['friction_points'])
        if set(values) != set(BRANCHES):
            continue
        lo,hi = values['minus'],values['plus']
        nominal = [values[b]['observed'] for b in ('nominal_a','nominal_b')]
        native = hi['observed'] - lo['observed']
        predicted = hi['predicted'] - lo['predicted']
        mode = ('cap_switch' if lo['cap_limited'] != hi['cap_limited'] else
                'cap' if lo['cap_limited'] else 'sticking')
        counts[mode] = counts.get(mode,0)+1
        rows.append(dict(lane=case['lane'], branches=values, formula_branch=mode,
                         native_signed_magnitude_mps=native, predicted_signed_magnitude_mps=predicted,
                         above_repeat_noise=abs(native) > max(1e-8,3*abs(nominal[1]-nominal[0]))))
    def rms(numbers):
        return math.sqrt(statistics.mean(x*x for x in numbers)) if numbers else None
    observed = [r['native_signed_magnitude_mps'] for r in rows]
    error = [r['predicted_signed_magnitude_mps']-v for r,v in zip(rows,observed,strict=True)]
    supported = [r for r in rows if r['above_repeat_noise']]
    return dict(stable_total=sum(c['mode']=='racket_stable' for c in cases), evaluated=len(rows),
                formula_branches=counts, signed_magnitude_rmse_mps=rms(error),
                zero_signed_magnitude_rmse_mps=rms(observed), above_repeat_noise=len(supported),
                sign_agreements=sum(r['native_signed_magnitude_mps']*r['predicted_signed_magnitude_mps']>0 for r in supported),
                cases=rows, scope='Magnitude of the net measured friction impulse divided by mass, compared with the scalar law at mean incident anchor speed and measured net normal load. Multiple anchor vectors can cancel.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','fit','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--role',choices=('training','development'),required=True)
    p.add_argument('--new-training-support',type=Path,help='Signed native support report for a separately registered training panel')
    a=p.parse_args();torch.set_num_threads(2)
    fit=json.loads(a.fit.read_text());status=json.loads((a.root/'status.json').read_text())
    expected=fit['g07_training']['status_sha256'] if a.role=='training' else fit['g07_sha256']
    if a.new_training_support:
        support=json.loads(a.new_training_support.read_text())
        if (a.role!='training' or not support['training_only'] or support['contact_receipt']
                or support['policy_training_admitted'] or support['physics_fit_sha256']!=sha(a.fit)):
            raise ValueError('Expected a training-only collection scored with this physics fit')
        expected=support['status_sha256']
    if a.role=='development' and a.root.name in ('half','quarter'):
        panel=a.root.parent
        scales=json.loads((panel/'completed.json').read_text())
        def identity(root,multiplier):
            startup=json.loads((root/'startup.json').read_text())
            spec=dict(startup['specs'][0]['spec'])
            return startup['source_sha256'],startup['seed'],spec.pop('epsilon_request')*multiplier,spec
        if (sha(panel/'wide'/'status.json')!=expected or scales['wide']!=expected or
                identity(panel/'wide',1)!=identity(a.root,2 if a.root.name=='half' else 4)):
            raise ValueError('Scale does not match held case and perturbation identity')
        expected=scales[a.root.name]
    if sha(a.root/'status.json')!=expected:raise ValueError('Native source changed')
    runs={}
    for item in status['completed']:
        folder=a.root/item['spec'];branch=item['branch']
        if sha(folder/branch/'native_probe.pt')!=item['probe_sha256']:raise ValueError('Changed native branch')
        runs[branch]=_load_branch(folder,branch)
    _same_contract(runs)
    phys=Path(__file__).resolve().parents[1]/'configs/shuttle_physics_v2.json'
    if sha(phys)!=fit['physics_sha256']:raise ValueError('Changed shuttle physics')
    mass=json.loads(phys.read_text())['mass_kg']
    lanes=runs['plus']['spec']['lanes']
    records={}; kinds={}
    geometry=BallImpulsePriorV1().preview
    for b in BRANCHES:
        rr=records[b]=[[] for _ in lanes]
        signs=[]
        for i,tr in enumerate(complete_window(runs[b]['raw'],runs[b]['spec'],160)):
            summarize_trace(tr,i,mass,rr,geometry);signs.append(signatures(tr))
        kinds[b]=[window_signature(v,160) for v in zip(*signs,strict=True)]
    cases=[]
    for col,lane in enumerate(lanes):
        mode=classify([kinds[b][col] for b in BRANCHES])
        raw={b:records[b][col] for b in BRANCHES}
        cases.append(dict(lane=lane,mode=mode,branches=raw,geometry=compare(raw['minus'],raw['plus'])))
    summary={}
    if a.new_training_support:
        from scripts.audit_supermimic_friction_reversal_support import compare as reversal_summary, single_load
        native={r['lane']:r for r in support['native_racket']['cases']}
        if set(native)!=set(lanes):
            raise ValueError('Signed support and native history cover different lanes')
        rows=[]
        seed=runs['plus']['spec']['contact_training_panel']['seed']
        for rec in cases:
            row=native[rec['lane']]
            if row['mode']!=rec['mode']:
                raise ValueError('Native contact classification changed')
            rec['case_id']=row['case_id']
            branches=rec['branches']
            rec['loaded_frames_by_branch']={k:len(v) for k,v in branches.items()}
            rec['friction_without_reported_normal']=any(
                t['normal_points']==0 and t['friction_points']>0 for branch in branches.values() for t in branch)
            rows.append(dict(seed=seed,lane=rec['lane'],kind=rec['mode'],
                         above_repeat_noise=(row['executed_target_change_rad']>0 and
                         torch.tensor(row['native_delta_mps']).norm().item()>max(1e-8,3*row['repeat_noise_mps'])),
                         **{k:single_load(v) for k,v in branches.items()}))
        summary=dict(signed_slip=reversal_summary(rows),support_sha256=sha(a.new_training_support),
                     friction_without_reported_normal=sum(r['friction_without_reported_normal'] for r in cases),
                     multiple_loaded_frames=sum(max(r['loaded_frames_by_branch'].values())>1 for r in cases))
    with a.output.open('x') as f:
        json.dump(dict(role=a.role,source_sha256=expected,cases=cases,**summary,
            measured_slip_law=measured_slip_law(cases,fit['selected']['friction'],fit['selected']['sticking']),
            scope='Raw sensor loads, pre/post twists and geometry. Anchor matching estimates continuity; contact IDs and solver stick/slip state are not observed.'),f,indent=2,allow_nan=False)


if __name__=='__main__':main()
