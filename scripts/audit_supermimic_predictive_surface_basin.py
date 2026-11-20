#!/usr/bin/env python3
"""Test a pre-impact, model-only basin selector against independent native finite differences."""
import argparse
import json
from pathlib import Path

import torch
from quadsmash.rl.supermimic_fixture_coordinates import fixture_court_origin
import torch.nn.functional as F

from audit_supermimic_surface_derivative_scales import vector_summary, read_material
from evaluate_supermimic_contact_rollout import BallImpulsePriorV1, sha, tick


def motion_from_initial(center,quat,velocity,omega,times):
    # Constant world twist of the link origin over one short diagnostic control interval.
    times=times[:,None,None]
    half=.5*times*omega.norm(dim=-1,keepdim=True)
    dxyz=.5*times*omega*torch.sinc(half/torch.pi)
    dw=half.cos()
    w=dw*quat[None,:,:1]-(dxyz*quat[None,:,1:]).sum(-1,keepdim=True)
    xyz=dw*quat[None,:,1:]+quat[None,:,:1]*dxyz+torch.linalg.cross(dxyz,quat[None,:,1:])
    return [center+times*velocity,torch.cat((w,xyz),-1),velocity.expand(len(times),-1,-1),omega.expand(len(times),-1,-1)]


def stable_prediction(events,speeds,eps):
    nominal=events[0]
    stable=((nominal['racket']>=0).sum(0)==1)&~nominal['ground'].any(0)&~nominal['net'].any(0)
    for event in events[1:]:
        for name in nominal:
            same = (nominal[name] == event[name]).reshape(*nominal[name].shape[:2],-1)
            stable &= same.all(-1).all(0)
    local=stable.clone()
    responses=[(speeds[i+1]-speeds[i])/(2*e) for e,i in zip(eps,(1,3,5))]
    left=(speeds[0]-speeds[5])/eps[-1];right=(speeds[6]-speeds[0])/eps[-1]
    for x,y in [(responses[0],responses[1]),(responses[1],responses[2]),(left,right)]:
        ratio=y.norm(dim=-1)/x.norm(dim=-1).clamp_min(1.e-12)
        local&=(F.cosine_similarity(x,y)>=.8)&(ratio>=.5)&(ratio<=2.)
    local&=responses[0].norm(dim=-1)*2*eps[0]>.01
    return stable,local,responses[0]


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('wide','labels','wide-audit','calibration','output'):
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--motion-readout',type=Path,help='Frozen unloaded-racket motion model fitted to calibration families only')
    ap.add_argument('--compound-material',type=Path)
    ap.add_argument('--material-readout',type=Path)
    a=ap.parse_args();torch.set_num_threads(3)
    if a.material_readout and not a.compound_material:
        ap.error('A material readout requires the native shape contract')
    audit=json.loads(a.wide_audit.read_text());labels=json.loads(a.labels.read_text())
    if audit['split'] not in ('train','development') or labels['wide_audit_sha256']!=sha(a.wide_audit) or labels['split']!=audit['split']:
        raise ValueError('Expected a matching calibration or development native panel')
    cfg=json.loads(a.calibration.read_text())
    if labels['calibration_sha256']!=sha(a.calibration):raise ValueError('Solver changed')
    rows=sorted(audit['samples'],key=lambda r:(r['seed'],r['lane']))
    if labels['case_ids']!=[[r['seed'],r['lane']] for r in rows]: raise ValueError('Different native cases or labels')
    prior=BallImpulsePriorV1().eval()
    eps=[.05,.025,.0125]
    times=torch.arange(prior.steps)*prior.sub_dt
    motion = None
    if a.motion_readout:
        from fit_supermimic_unloaded_racket_motion import motion_features, predict_motion
        prov = json.loads((a.wide/'nominal_a'/'provenance.json').read_text())
        teacher_path = Path(prov['args'][prov['args'].index('--teacher')+1])
        motion = torch.load(a.motion_readout, weights_only=True)
        if (motion['schema'] != 'unloaded-articulation-motion-readout-v1'
                or motion['design_sha256'] != sha(a.wide/'design.json')
                or motion['teacher_sha256'] != sha(teacher_path) or motion['physics_dt_s'] != prior.sub_dt):
            raise ValueError('Mismatch between motion model and validation fixture')
        teacher = torch.load(teacher_path, weights_only=True, mmap=True)
    prov=json.loads((a.wide/'nominal_a'/'provenance.json').read_text())
    material=read_material(a.wide,prov,a.compound_material,a.material_readout,a.calibration)
    material_sha=sha(a.compound_material) if material is not None else None
    readout_sha=sha(a.material_readout) if a.material_readout else None
    if material_sha!=labels.get('material_sha256') or readout_sha!=labels.get('coefficient_fit_sha256'):
        raise ValueError('Different contact transition in derivative labels')
    events=[];sufficient=[];pred=[]
    for seed in sorted({r['seed'] for r in rows}):
        # Only a single branch is opened: all seven trajectories start from this same native state.
        path=a.wide/'nominal_a'/f'spec_{seed}'/'trace.pt'
        if sha(path)!=audit['archives'][str(path)]:raise ValueError('Parent nominal trace changed')
        src=torch.load(path,weights_only=True,mmap=True)
        raw=src['raw']['substep_trace']
        if raw['contact_metadata'].get('initial_racket_twist_source')!='direct_native_link_velocity_after_intervention':
            raise ValueError('Requires direct initial native surface twist')
        t=raw['tensors'];s=t['shuttle_state_w'][0]
        ball=torch.cat((s[:,:3],s[:,7:10]),dim=-1)
        c,q,v,omega=[t[k][0] for k in ('racket_position_w','racket_quaternion_w','racket_linear_velocity_w','racket_angular_velocity_w')]
        origin=fixture_court_origin(src,c,json.loads((a.wide/'design.json').read_text()))
        c=c-origin
        ball[:,:3]-=origin
        direction=src['velocity_intervention']['direction_world']
        features = motion_features(src, teacher) if motion is not None else None
        speeds=[];signatures=[]
        for e in (0.,-eps[0],eps[0],-eps[1],eps[1],-eps[2],eps[2]):
            if motion is None:
                stream=motion_from_initial(c,q,v+e*direction,omega,times)
            else:
                stream=predict_motion(motion,features,[c,q,v+e*direction,omega],times)
            sig={}
            out,_=tick(prior,ball,[x[0] for x in stream],[x[-1] for x in stream],
                       cfg['fitted'],cfg['solver']['pair_contact_offset_m'],
                       None if material is not None else torch.zeros(len(ball),dtype=torch.bool),
                       measured_path=stream,event_trace=sig,racket_material=material,
                       initial_spin=s[:,10:13] if material is not None else None,compound_racket=material is not None)
            speeds.append(out[:,3:]);signatures.append(sig)
        stable,qualified,d=stable_prediction(signatures,speeds,eps)
        events.append(stable);sufficient.append(qualified);pred.append(d)
    native=torch.tensor([r['delta']['native'] for r in rows])/(2*eps[0])
    model=torch.cat(pred);all_rows=torch.ones(len(rows),dtype=torch.bool)
    true=torch.tensor(labels['incident_labels']['native_convergent_stable'])
    types=[r['kind'] for r in rows]
    groups={}
    for name,mask in [('all',all_rows),('predicted_event_stable',torch.cat(events)),('predicted_convergent',torch.cat(sufficient)),('native_convergent_diagnostic',true)]:
        g=dict(rows=int(mask.sum()),native_convergent=int((true&mask).sum()),
               false_positive_no_native_racket=sum(t=='no_contact' and bool(x) for t,x in zip(types,mask)))
        if mask.any():g['signed']=vector_summary(native[mask],model[mask])
        groups[name]=g
    with a.output.open('x') as f:
        json.dump(dict(scope='Pre-impact state and prescribed perturbation only; hyperparameters fixed by physical convergence diagnostic',
                       material_sha256=material_sha,coefficient_fit_sha256=readout_sha,
                       motion_readout_sha256=sha(a.motion_readout) if motion is not None else None,groups=groups,labels_sha256=sha(a.labels),calibration_sha256=sha(a.calibration),
                       contact_bptt_admitted=False,actor_updates=0),f,indent=2,allow_nan=False)


if __name__=='__main__':main()
