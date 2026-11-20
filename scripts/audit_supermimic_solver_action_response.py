#!/usr/bin/env python3
"""Differentiate the calibrated contact rollout along actual paired command sequences."""
import argparse
import json
from pathlib import Path

import torch

from audit_supermimic_paired_body_transfer import response_metrics
from evaluate_supermimic_contact_rollout import BallImpulsePriorV1, PhysicalWorldV2, racket, sha, tick
from quadsmash.rl.supermimic_action_identification_dataset_v1 import _load_branch, _same_contract
from quadsmash.rl.supermimic_native_pair_capture_v1 import BRANCHES


def sphere_tangent_from_spin(spin, normal_force, radius):
    """Measured-spin oracle, assuming a single contact at minus radius times the normal."""
    delta_spin = torch.diff(spin,dim=0)
    normals = torch.nn.functional.normalize(normal_force[1:],dim=-1)
    # I/(m*r) = 2*r/5 for the authored solid sphere. Opposite signs of
    # the two cross products in the angular and linear relations cancel.
    return (0.4*radius*torch.cross(normals,delta_spin,dim=-1)).sum(dim=0)


def tangent_from_contact_arm(relative, arm, normal_dv, radius, friction):
    """Tangential no-slip impulse with a fixed moving surface and a solid sphere.

    J/m is in m/s. An off-centre normal impulse also changes the tangential
    velocity of the contact point through its angular response.
    """
    n = torch.nn.functional.normalize(normal_dv, dim=-1)
    eye = torch.eye(3, dtype=arm.dtype, device=arm.device)
    projection = eye - n[..., :, None]*n[..., None, :]
    inverse_mass = eye + (arm.square().sum(-1)[..., None, None]*eye
                          - arm[..., :, None]*arm[..., None, :]) / (.4*radius**2)
    # The normal coordinate is fixed to zero in the tangent solve.
    system = projection @ inverse_mass @ projection + n[..., :, None]*n[..., None, :]
    rhs = projection @ (relative[..., None] + inverse_mass @ normal_dv[..., None])
    sticking = -torch.linalg.solve(system, rhs).squeeze(-1)
    limit = friction*normal_dv.norm(dim=-1,keepdim=True)
    return sticking * (limit/sticking.norm(dim=-1,keepdim=True).clamp_min(1e-12)).clamp(max=1.)


def tangent_at_measured_contact(trace, radius, normal_dv, friction, *, measured_arm=False):
    """Predict slip at measured force frames using the incoming sphere and racket velocities."""
    normal = torch.nn.functional.normalize(normal_dv,dim=-1)
    shuttle = trace['shuttle_state_w'][:-1]
    arm = -radius*normal
    point = trace['racket_contact_position_w'][1:]
    point = torch.where(torch.isfinite(point).all(-1,keepdim=True),point,shuttle[...,:3]+arm)
    if measured_arm:
        arm = point-shuttle[...,:3]
    sphere_velocity = shuttle[...,7:10] + torch.cross(shuttle[...,10:13],arm,dim=-1)
    racket_arm = point-trace['racket_position_w'][:-1]
    surface = trace['racket_linear_velocity_w'][:-1] + torch.cross(
        trace['racket_angular_velocity_w'][:-1],racket_arm,dim=-1)
    relative = sphere_velocity-surface
    if measured_arm:
        return tangent_from_contact_arm(relative, arm, normal_dv, radius, friction).sum(0)
    slip = relative-(relative*normal).sum(-1,keepdim=True)*normal
    magnitude = torch.minimum(2/7*slip.norm(dim=-1),friction*normal_dv.norm(dim=-1))
    return -(magnitude[...,None]*torch.nn.functional.normalize(slip,dim=-1)).sum(0)


def mode(run,lane):
    raw=run['raw'];tr=raw['substep_trace']['tensors']
    contact=tr['racket_contact_position_valid'][:,lane].flatten().bool()
    first=contact.nonzero().flatten()
    intervals=int(contact[0])+int((~contact[:-1]&contact[1:]).sum())
    flags=tuple(raw['task_after'][name][lane].item() for name in
                ('had_contact','released','frame_hit','string_hit','contact_count'))
    return (flags,intervals,int(first[0]) if len(first) else -1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('dataset','body','calibration','output'):
        p.add_argument('--'+k,required=True,type=Path)
    a=p.parse_args();torch.set_num_threads(3)
    ds=torch.load(a.dataset,map_location='cpu',weights_only=True)
    ck=torch.load(a.body,map_location='cpu',weights_only=True)
    folder=Path(ds['source_collection'])
    if sha(folder/'status.json')!=ds['source_status_sha256']:
        raise ValueError('Collection fingerprint changed')
    hashes={(x['spec'],x['branch']):x['probe_sha256'] for x in json.loads((folder/'status.json').read_text())['completed']}
    task=ck['source_manifest']['task_contract'];cfg=json.loads(a.calibration.read_text())
    body=PhysicalWorldV2(task['asset']['sha256']).eval().requires_grad_(False)
    body.load_state_dict(ck['model'])
    prior=BallImpulsePriorV1().eval().requires_grad_(False)
    rows=ds['rows'];lanes=[r['lane_index'] for r in rows]
    if set(lanes)&set(ck['calibration']['train_lanes']):raise ValueError('Overlapping action-ID and fit lanes')
    modes={}
    traces={b:[] for b in ('minus','plus')}
    for name in sorted(set(r['spec_name'] for r in rows)):
        runs={k:_load_branch(folder/name,k) for k in BRANCHES}
        _same_contract(runs)
        for b,run in runs.items():
            if task!=run['task_contract'] or sha(folder/name/b/'native_probe.pt')!=hashes[name,b]:
                raise ValueError('Probe source or robot identity changed')
        for row in (r for r in rows if r['spec_name']==name):
            lane=row['lane_index']
            for b in traces:
                trace=runs[b]['raw']['substep_trace']
                if abs(trace['physics_dt_s']-prior.sub_dt)>1e-12 or trace['count']!=prior.steps+1:
                    raise ValueError('Event-tick substep coverage differs from the solver timebase')
                t=trace['tensors']
                fields = ['shuttle_state_w','racket_position_w','racket_quaternion_w',
                          'racket_linear_velocity_w','racket_angular_velocity_w','racket_force_w','racket_contact_position_w']
                if 'racket_friction_force_w' in t:
                    fields.append('racket_friction_force_w')
                traces[b].append((row['sample_id'], {key:t[key][:,lane].clone() for key in fields}))
            branches=[mode(runs[b],lane) for b in BRANCHES]
            if branches[0]!=branches[1]:group='nominal_mismatch'
            elif any(b[:2]!=branches[0][:2] for b in branches):group='contact_switch'
            elif len({b[2] for b in branches})>1:group='toi_switch'
            elif branches[0][2]<0:group='no_contact'
            else:group='stable'
            modes[name,lane]=dict(group=group,branch_events=dict(zip(BRANCHES,branches)))
    def f(key):return torch.stack([r[key] for r in rows])
    sp=Path(task['suite']['path'])
    if sha(sp)!=task['suite']['sha256']:raise ValueError('Different launch suite')
    launch=torch.tensor([row['launch']['position_m']+row['launch']['velocity_mps'] for row in
                        json.loads(sp.read_text())['cases']])[lanes]
    config_path=Path(__file__).resolve().parents[1]/'configs/shuttle_physics_v2.json'
    shuttle=json.loads(config_path.read_text())
    mass=shuttle['mass_kg']
    native_terms={}
    paths={}
    friction_available=all('racket_friction_force_w' in entry for records in traces.values()
                           for _,entry in records)
    for b,records in traces.items():
        lookup=dict(records)
        ordered=[lookup[row['sample_id']] for row in rows]
        paths[b]={key:torch.stack([entry[key] for entry in ordered],1) for key in ordered[0]
                  if key!='racket_friction_force_w' or friction_available}
        t=paths[b]
        vel=t['shuttle_state_w'][...,7:10]
        rel=vel[:-1]-prior.wind
        flight=(prior.sub_dt*(prior.gravity-prior.drag*rel.norm(dim=-1,keepdim=True)*rel)).sum(0)
        normal=t['racket_force_w'][1:].sum(0)*(prior.sub_dt/mass)
        extra=(vel[-1]-vel[0])-flight-normal
        native_terms[b]=dict(incoming=vel[0],flight=flight,racket_normal=normal,unexplained=extra,
            spin_at_start=t['shuttle_state_w'][0,:,10:13].norm(dim=-1))
        native_terms[b]['spin_oracle_tangent'] = sphere_tangent_from_spin(
            t['shuttle_state_w'][...,10:13], t['racket_force_w'], shuttle['collision_radius_m'])
        native_terms[b]['slip_at_measured_contact'] = tangent_at_measured_contact(
            t,shuttle['collision_radius_m'],t['racket_force_w'][1:]*(prior.sub_dt/mass),cfg['fitted']['friction'])
        native_terms[b]['measured_arm_tangent'] = tangent_at_measured_contact(
            t,shuttle['collision_radius_m'],t['racket_force_w'][1:]*(prior.sub_dt/mass),
            cfg['fitted']['friction'],measured_arm=True)
        if friction_available:
            friction=t['racket_friction_force_w'][1:].sum(0)*(prior.sub_dt/mass)
            native_terms[b].update(racket_friction=friction,unexplained_after_friction=extra-friction)
        native_ball=f('step_next_states_'+b)[:,-1,49:]
        end=paths[b]['shuttle_state_w'][-1]
        torch.testing.assert_close(native_ball, torch.cat((end[:,:3],end[:,7:10]),-1),atol=1e-5,rtol=1e-5)
    t0,t1=f('step_targets_minus'),f('step_targets_plus')
    u0,u1=f('step_final_actions_minus'),f('step_final_actions_plus')
    if not bool(f('present').all()):raise ValueError('Contact-only evaluator requires ball in flight')
    def forward(alpha,teacher=None,*,event_path=None,native_event_start=False):
        targets=torch.lerp(t0,t1,alpha[:,None,None])
        actions=torch.lerp(u0,u1,alpha[:,None,None])
        s=f('state');prev=f('previous_action');feet=f('foot_contact').float()
        time=f('elapsed');active=f('present');patch=torch.zeros_like(active).bool()
        events=[]
        for i in range(targets.shape[1]):
            robot,logits,_=body(s,targets[:,i],prev,feet,active,time,launch)
            if teacher is not None:robot=teacher[:,i]
            start_ball=s[:,49:]
            measured=None
            if i+1==targets.shape[1] and event_path is not None:
                measured=[event_path[k][:-1] for k in ('racket_position_w','racket_quaternion_w',
                                                        'racket_linear_velocity_w','racket_angular_velocity_w')]
                if native_event_start:
                    initial=event_path['shuttle_state_w'][0]
                    start_ball=torch.cat((initial[:,:3],initial[:,7:10]),-1)
            ball,hit=tick(prior,start_ball,racket(s,body.geometry),racket(robot,body.geometry),
                          cfg['fitted'],cfg['solver']['pair_contact_offset_m'],patch,measured_path=measured)
            events.append(hit);s=torch.cat((robot[:,:49],ball),-1)
            prev=actions[:,i];feet=logits.sigmoid();time=time+prior.dt
        return s[:,52:],torch.stack(events,1)
    alpha=t0.new_full((len(rows),),.5,requires_grad=True)
    mid,_=forward(alpha)
    jac=torch.stack([torch.autograd.grad(mid[:,i].sum(),alpha,retain_graph=i<2)[0] for i in range(3)],-1)
    native={b:f('step_next_states_'+b) for b in BRANCHES}
    delta=native['plus'][:,-1,52:]-native['minus'][:,-1,52:]
    repeat=native['nominal_b'][:,-1,52:]-native['nominal_a'][:,-1,52:]
    with torch.no_grad():
        minus,hm=forward(torch.zeros_like(alpha));plus,hp=forward(torch.ones_like(alpha))
        oracle0,_=forward(torch.zeros_like(alpha),native['minus']);oracle1,_=forward(torch.ones_like(alpha),native['plus'])
    predictions={'body_endpoint_difference':plus-minus,'local_sequence_jacobian':jac.detach(),
                 'native_robot_endpoint_difference':oracle1-oracle0}
    with torch.no_grad():
        for native_start in (False,True):
            b0,_=forward(torch.zeros_like(alpha),native['minus'],event_path=paths['minus'],native_event_start=native_start)
            b1,_=forward(torch.ones_like(alpha),native['plus'],event_path=paths['plus'],native_event_start=native_start)
            key='measured_substeps_with_native_event_start' if native_start else 'measured_substeps_with_predicted_ball'
            predictions[key]=b1-b0
    parts={key:native_terms['plus'][key]-native_terms['minus'][key] for key in
           ('incoming','flight','racket_normal','unexplained')}
    torch.testing.assert_close(sum(parts.values()),delta,atol=1e-5,rtol=1e-5)
    predictions['native_incoming_plus_flight_and_reported_normal']=parts['incoming']+parts['flight']+parts['racket_normal']
    parts['spin_oracle_tangent'] = native_terms['plus']['spin_oracle_tangent']-native_terms['minus']['spin_oracle_tangent']
    predictions['native_normal_and_measured_spin_oracle'] = (
        predictions['native_incoming_plus_flight_and_reported_normal'] + parts['spin_oracle_tangent'])
    parts['slip_at_measured_contact'] = native_terms['plus']['slip_at_measured_contact']-native_terms['minus']['slip_at_measured_contact']
    predictions['native_normal_and_incoming_slip_model'] = (
        predictions['native_incoming_plus_flight_and_reported_normal'] + parts['slip_at_measured_contact'])
    parts['measured_arm_tangent'] = native_terms['plus']['measured_arm_tangent']-native_terms['minus']['measured_arm_tangent']
    predictions['native_normal_and_measured_arm_model'] = (
        predictions['native_incoming_plus_flight_and_reported_normal'] + parts['measured_arm_tangent'])
    if friction_available:
        for key in ('racket_friction','unexplained_after_friction'):
            parts[key] = native_terms['plus'][key]-native_terms['minus'][key]
        with_friction = predictions['native_incoming_plus_flight_and_reported_normal']+parts['racket_friction']
        torch.testing.assert_close(with_friction+parts['unexplained_after_friction'],delta,atol=1e-5,rtol=1e-5)
        predictions['native_incoming_plus_flight_and_reported_racket_forces']=with_friction
    tags=[modes[r['spec_name'],r['lane_index']]['group'] for r in rows]
    groups={}
    for name in ['all']+sorted(set(tags)):
        mask=torch.tensor([name=='all' or name==t for t in tags])
        groups[name]={k:response_metrics(delta[mask],v[mask],repeat[mask]) for k,v in predictions.items()}
    details=[]
    for i,row in enumerate(rows):
        details.append(dict(case=row['sample_id'],native_delta=delta[i].tolist(),repeat_delta=repeat[i].tolist(),
                            model_minus_contact_ticks=hm[i].tolist(),model_plus_contact_ticks=hp[i].tolist(),
                            native_event_contributions={k:v[i].tolist() for k,v in parts.items()},
                            native_branch_diagnostics={b:{k:v[i].tolist() for k,v in native_terms[b].items()
                                if k == 'spin_at_start'} for b in ('minus','plus')},
                            **modes[row['spec_name'],row['lane_index']],**{k:v[i].tolist() for k,v in predictions.items()}))
    result=dict(scope='paired measured final target and action history sequences, control-step pose interpolation; native substep timing available at final intervention tick only',
                sources={k:sha(getattr(a,k)) for k in ('dataset','body','calibration')},
                validation=groups,samples=details,all_signed_traces_include_friction=friction_available,contact_bptt_admitted=False,actor_updates=0)
    with a.output.open('x') as output:json.dump(result,output,indent=2,allow_nan=False)


if __name__=='__main__':
    main()
