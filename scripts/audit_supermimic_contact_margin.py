#!/usr/bin/env python3
"""Estimate contact derivative reliability from pre-impact physical margins, with launch-family cross-validation."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from audit_supermimic_predictive_surface_basin import motion_from_initial, stable_prediction
from audit_supermimic_surface_derivative_scales import vector_summary
from evaluate_supermimic_contact_rollout import BallImpulsePriorV1, sha, tick
from quadsmash.rl.supermimic_ball_geometry_preview_v1 import box_gap, round_gap
from quadsmash.rl.supermimic_contact_pilot_v1 import _quat_rotate, _quat_rotate_inverse
from quadsmash.rl.supermimic_fixture_coordinates import fixture_court_origin


NAMES = ('surface_gap_margin_m', 'toi_phase_margin', 'normal_speed_mps', 'tangent_speed_mps',
         'perturbation_normal_alignment', 'face_normal_alignment', 'surface_angular_speed_radps',
         'time_to_impact_s', 'incidence_cosine', 'predicted_response_gain')


def margins(prior, ball, motion, events, direction, derivative, times):
    c,q,v,w = motion
    index = (events['racket'] >= 0).int().argmax(0)
    lane = torch.arange(len(ball))
    pos,vel = prior.advance_trajectory(ball[:,:3],ball[:,3:])
    point = pos[lane,index]
    local = _quat_rotate_inverse(q[index,lane],point-c[index,lane])
    geom = prior.preview
    distances = torch.cat((box_gap(local,geom.racket_box_centers,geom.racket_box_halves),
        round_gap(local,geom.racket_round_centers,geom.racket_round_axes,geom.racket_round_halves,
                  geom.racket_round_radii,geom.racket_round_capsule)),-1) - geom.radius
    two = distances.topk(2,largest=False).values
    local_normal,_ = prior.racket_surface(local)
    normal = _quat_rotate(q[index,lane],local_normal)
    arm = point-c[index,lane]-geom.radius*normal
    relative = vel[lane,index]-v[index,lane]-torch.linalg.cross(w[index,lane],arm)
    vn = (relative*normal).sum(-1)
    tangential = (relative - vn[:,None]*normal).norm(dim=-1)
    phase = (two[:,0]/(-vn*prior.sub_dt).clamp_min(1e-6)).clamp(0,1)
    return torch.stack((two[:,1]-two[:,0], torch.minimum(phase,1-phase), -vn, tangential,
                        (normal*direction).sum(-1).abs(),local_normal[:,0].abs(),w[index,lane].norm(dim=-1),
                        times[index],(-vn)/relative.norm(dim=-1).clamp_min(1e-6),derivative.norm(dim=-1)),-1)


@torch.no_grad()
def observed_split(wide, labels_path, split, cfg, prior):
    design=json.loads((wide/'design.json').read_text())
    audit_file=wide/(split+'.json')
    audit=json.loads(audit_file.read_text())
    labels=json.loads(labels_path.read_text())
    if (design.get('expected_origin_spacing_m') != 0. or labels['wide_audit_sha256'] != sha(audit_file)
            or labels['split'] != split or audit['design_sha256'] != sha(wide/'design.json')):
        raise ValueError('Requires the current fixed zero-origin multiscale panel')
    rows=sorted(audit['samples'],key=lambda r:(r['seed'],r['lane']))
    if labels['case_ids'] != [[r['seed'],r['lane']] for r in rows]:
        raise ValueError('Mismatched physical cases')
    eps=[.05,.025,.0125]
    times=torch.arange(prior.steps)*prior.sub_dt
    features=[];eligible=[];predictions=[]
    for seed in sorted({r['seed'] for r in rows}):
        path=wide/'nominal_a'/f'spec_{seed}'/'trace.pt'
        if sha(path)!=audit['archives'][str(path)]:
            raise ValueError('Changed nominal case')
        packet=torch.load(path,mmap=True,weights_only=True)
        tr=packet['raw']['substep_trace']['tensors']
        c,q,v,w=[tr[k][0] for k in ('racket_position_w','racket_quaternion_w','racket_linear_velocity_w','racket_angular_velocity_w')]
        origin=fixture_court_origin(packet,c,design)
        s=tr['shuttle_state_w'][0]
        ball=torch.cat((s[:,:3]-origin,s[:,7:10]),-1)
        direction=packet['velocity_intervention']['direction_world']
        signatures=[];speeds=[]
        for e in (0.,-.05,.05,-.025,.025,-.0125,.0125):
            motion=motion_from_initial(c-origin,q,v+e*direction,w,times)
            sig={}
            out,_=tick(prior,ball,[x[0] for x in motion],[x[-1] for x in motion],cfg['fitted'],
                       cfg['solver']['pair_contact_offset_m'],torch.zeros(len(c),dtype=torch.bool),
                       measured_path=motion,event_trace=sig)
            if e==0.:
                nominal=motion
            signatures.append(sig);speeds.append(out[:,3:])
        _,retain,pred=stable_prediction(signatures,speeds,eps)
        features.append(margins(prior,ball,nominal,signatures[0],direction,pred,times))
        eligible.append(retain);predictions.append(pred)
    native=torch.tensor([r['delta']['native'] for r in rows])/(2*eps[0])
    pred=torch.cat(predictions); std=labels['incident_labels']['native_convergent_stable']
    gain=pred.norm(dim=-1)/native.norm(dim=-1).clamp_min(1.e-7)
    target=torch.tensor(std)&(F.cosine_similarity(pred,native)>=.5)&(gain>=.5)&(gain<=2)
    return dict(features=torch.cat(features), support=torch.cat(eligible),target=target,native=native,
                pred=pred,convergent=torch.tensor(std),family=torch.tensor([r['seed'] for r in rows]))


@torch.enable_grad()
def fit(x,y):
    if len(x) < 2 or y.all() or not y.any():
        raise ValueError('Insufficient physical support to fit margin classifier')
    mean=x.mean(0);scale=x.std(0).clamp_min(1.e-7)
    x=((x-mean)/scale).clamp(-6,6)
    w=torch.zeros(x.shape[1],requires_grad=True)
    b=y.float().mean().logit().detach().requires_grad_()
    opt=torch.optim.LBFGS([w,b],max_iter=60,line_search_fn='strong_wolfe')
    def loss():
        opt.zero_grad()
        value=F.binary_cross_entropy_with_logits(x@w+b,y.float())+.025*w.square().sum()
        value.backward()
        return value
    opt.step(loss)
    return dict(mean=mean,scale=scale,w=w.detach(),b=b.detach())


def predict(model,x):
    return ((((x-model['mean'])/model['scale']).clamp(-6,6))@model['w']+model['b']).sigmoid()


def stats(d,mask):
    n=int(mask.sum())
    r=dict(rows=n,families=int(d['family'][mask].unique().numel()),
           native_convergent=int(d['convergent'][mask].sum()),direction_and_gain_correct=int(d['target'][mask].sum()))
    if n:r['response']=vector_summary(d['native'][mask],d['pred'][mask])
    return r


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ('panel','calibration','output'):
        cli.add_argument('--'+name,type=Path,required=True)
    a=cli.parse_args();torch.set_num_threads(3)
    cfg=json.loads(a.calibration.read_text());prior=BallImpulsePriorV1().eval()
    def load(split):
        path=a.panel/'scales'/(split+'.json')
        if json.loads(path.read_text())['calibration_sha256']!=sha(a.calibration):
            raise ValueError('Solver changed after pairing')
        return observed_split(a.panel/'wide',path,split,cfg,prior)
    train=load('train')
    mask=train['support'];family=train['family'];x=train['features'];y=train['target']
    cross=torch.zeros_like(y,dtype=torch.float)
    for held in family.unique():
        fit_mask=mask&(family!=held);held_mask=mask&(family==held)
        cross[held_mask]=predict(fit(x[fit_mask],y[fit_mask]),x[held_mask])
    # Require a trustworthy calibration transfer across families before unlocking any selector.
    grid=[]
    for threshold in torch.arange(.05,1.,.05):
        selected=mask&(cross>=threshold)
        scores=stats(train,selected)
        grid.append(dict(threshold=float(threshold),**scores))
    valid=[v for v in grid if v['rows']>=8 and v['families']>=2 and v['direction_and_gain_correct']>=.9*v['rows']]
    selected=min(valid,key=lambda v:v['threshold'])['threshold'] if valid else None
    model=fit(x[mask],y[mask])
    dev=load('development');prob=predict(model,dev['features'])
    # Report ranking even when the calibration cut has insufficient support.
    rankings={}
    for quantile in (.5,.75,.9):
        cut=float(cross[mask].quantile(quantile))
        rankings[str(quantile)]=dict(training_probability_cut=cut,development=stats(dev,dev['support']&(prob>=cut)))
    report=dict(scope='Initial state, prescribed perturbation and causal surface model only; learned on calibration families',
                features=NAMES,weights=model['w'].tolist(),intercept=float(model['b']),input_mean=model['mean'].tolist(),
                input_scale=model['scale'].tolist(),calibration_sha256=sha(a.calibration),
                label_sha256={k:sha(a.panel/'scales'/(k+'.json')) for k in ('train','development')},
                oof=grid,selected_training_threshold=selected,development_eligible=stats(dev,dev['support']),
                development_selected=stats(dev,dev['support']&(prob>=selected)) if selected is not None else None,
                exploratory_ranking=rankings,policy_updates=0,contact_gradients_admitted=False)
    with a.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)


if __name__=='__main__':main()
