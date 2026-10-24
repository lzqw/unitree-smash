#!/usr/bin/env python3
"""CPU free-flight shooting proposals from immutable TRAIN public observations.

These are planning calculations, not native hits, feasibility proofs, or scores.
The full-racket control and the frozen success thresholds are never changed.
"""
import argparse
from datetime import datetime,timezone
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from quadsmash.control.initial_intercept import predict_initial_intercept
from quadsmash.physics.court_v4 import segment_top


def free_flight(position,velocity,physics,court,dt=.002):
    """RK4 flight under the existing quadratic drag; no collision responses."""
    position=np.asarray(position,dtype=float).copy();velocity=np.asarray(velocity,dtype=float).copy()
    count=len(position);landed=np.zeros(count,dtype=bool);crossed=landed.copy()
    landing=np.full_like(position,np.nan);crossing=landing.copy();duration=np.full(count,np.nan)
    radius=physics['collision_radius_m'];plane=court['net_plane_x_m']+radius+court['net_equivalent_thickness_m']/2
    wind=np.array(physics['wind_m_s']);drag=physics['quadratic_drag_k_per_m']
    def acceleration(v):
        relative=v-wind
        return np.array([0.,0.,-9.81])-drag*np.linalg.norm(relative,axis=1,keepdims=True)*relative
    for step in range(int(np.ceil(1.5/dt))):
        a1=acceleration(velocity);v2=velocity+.5*dt*a1;a2=acceleration(v2)
        v3=velocity+.5*dt*a2;a3=acceleration(v3);v4=velocity+dt*a3;a4=acceleration(v4)
        next_position=position+dt*(velocity+2*v2+2*v3+v4)/6
        next_velocity=velocity+dt*(a1+2*a2+2*a3+a4)/6
        hit=(~crossed)&(~landed)&(position[:,0]<=plane)&(next_position[:,0]>=plane)
        alpha=np.clip((plane-position[:,0])/np.maximum(next_position[:,0]-position[:,0],1e-12),0.,1.)
        crossing[hit]=(position+alpha[:,None]*(next_position-position))[hit];crossed|=hit
        hit=(~landed)&(position[:,2]>radius)&(next_position[:,2]<=radius)
        alpha=np.clip((position[:,2]-radius)/np.maximum(position[:,2]-next_position[:,2],1e-12),0.,1.)
        landing[hit]=(position+alpha[:,None]*(next_position-position))[hit]
        duration[hit]=(step+alpha[hit])*dt;landed|=hit
        position,velocity=next_position,next_velocity
        if landed.all():break
    half=court['net_width_m']/2;n=court['net_profile_segments']
    band_tops=np.array([segment_top(i,court) for i in range(n)])
    # A conservative aim margin over the highest strip touched laterally by
    # the finite-radius sphere, evaluated when its trailing side clears net.
    y=np.nan_to_num(crossing[:,1]);a=np.clip(((y-radius+half)*n/(2*half)).astype(int),0,n-1)
    b=np.clip(((y+radius+half)*n/(2*half)).astype(int),0,n-1)
    height=np.maximum(band_tops[a],band_tops[b]);margin=crossing[:,2]-radius-height
    return dict(landing_m=landing,crossing_m=crossing,net_bottom_margin_m=margin,duration_s=duration)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run=a.run.resolve();output=a.output.resolve()
    if output.exists():raise ValueError('Use a fresh proposal output')
    physics=json.loads((ROOT/'configs/shuttle_physics_v2.json').read_text())
    court=json.loads((ROOT/'configs/court_net_v4.json').read_text())
    logs=json.loads((run/'causal_plans.partial.json').read_text())
    log=next(row for row in logs if 62 in row['env_ids']);i=log['env_ids'].index(62)
    public=np.asarray(log['used_public_observations'][i])[None]
    contacts=json.loads((run/'contacts_001.json').read_text())
    event=next(row for row in contacts if row['env_id']==62 and row['episode_index']==2 and row['first_racket_contact'])
    assert event['string_contact'] and event['legal_initial_contact']
    intercepts=np.array([.95,1.05,1.167252540588379,1.25,1.35,1.4])
    forecast=predict_initial_intercept(np.repeat(public,len(intercepts),axis=0),[1.10,0.,1.80],[1.,0.,0.],
        intercepts,physics,dt=.001,max_seconds=1.5)
    assert forecast['valid'].all()
    actual_incoming=predict_initial_intercept(public,[1.10,0.,1.80],[1.,0.,0.],
        event['ball_position_m'][0],physics,dt=.001,max_seconds=1.5)
    states=[dict(kind='Actual recorded ball center at native contact; incoming velocity is model forecast',
        position_m=event['ball_position_m'],time_s=event['time_s'],incoming_velocity_mps=actual_incoming['incoming_velocity_mps'][0].tolist())]
    states += [dict(kind='Causal TRAIN planning proposal, not an executed contact',position_m=position.tolist(),
        time_s=float(public[0,775]+clock),incoming_velocity_mps=velocity.tolist())
        for position,clock,velocity in zip(forecast['position_m'],forecast['arrival_time_s'],forecast['incoming_velocity_mps'])]
    vx,vz=np.meshgrid(np.arange(4.,12.0001,.1),np.arange(-.5,-7.0001,-.1))
    grid=np.column_stack((vx.ravel(),np.zeros(vx.size),vz.ravel()));samples=len(grid)
    initial=np.repeat(np.array([x['position_m'] for x in states]),samples,axis=0)
    outgoing=np.tile(grid,(len(states),1));incoming=np.repeat(np.array([x['incoming_velocity_mps'] for x in states]),samples,axis=0)
    result=free_flight(initial,outgoing,physics,court)
    speed=np.linalg.norm(outgoing,axis=1)
    valid=(speed>=8.)&(outgoing[:,0]>0)&(outgoing[:,2]<=-1)&(result['net_bottom_margin_m']>=.005)
    valid&=(result['landing_m'][:,0]>=court['target_x_limits_m'][0])&(result['landing_m'][:,0]<=court['target_x_limits_m'][1])
    valid&=(np.abs(result['landing_m'][:,1])<=court['target_y_half_width_m'])
    delta=outgoing-incoming;normal=delta/np.linalg.norm(delta,axis=1,keepdims=True)
    required_normal_speed=(incoming*normal).sum(1)+np.linalg.norm(delta,axis=1)/(1+physics['restitution'])
    selected=[]
    for index,state in enumerate(states):
        ids=np.flatnonzero(valid[index*samples:(index+1)*samples])+index*samples
        item=dict(state,candidate_count=samples,grid_candidates_meeting_idealized_attack_clearance_and_target=int(len(ids)))
        if len(ids):
            score=np.abs(required_normal_speed[ids])+.04*(speed[ids]-8.)+.05*np.abs(result['landing_m'][ids,0]-3.)
            best=int(ids[score.argmin()]);selected.append(best)
            item['proposal']=dict(outgoing_velocity_mps=outgoing[best].tolist(),outgoing_speed_mps=float(speed[best]),
                predicted_landing_m=result['landing_m'][best].tolist(),predicted_net_bottom_margin_m=float(result['net_bottom_margin_m'][best]),
                predicted_duration_s=float(result['duration_s'][best]),idealized_frictionless_face_normal_world=normal[best].tolist(),
                idealized_required_point_normal_speed_mps=float(required_normal_speed[best]))
        state.update(item)
    if selected:
        fine=free_flight(initial[selected],outgoing[selected],physics,court,dt=.001)
        landing_error=np.linalg.norm(fine['landing_m']-result['landing_m'][selected],axis=1)
        margin_error=np.abs(fine['net_bottom_margin_m']-result['net_bottom_margin_m'][selected])
        assert landing_error.max()<.0005 and margin_error.max()<.0005
    else:landing_error=margin_error=np.array([np.nan])
    # Constant-acceleration analytical check, independent of event interpolation.
    no_drag=dict(physics,quadratic_drag_k_per_m=0.)
    analytic=free_flight(np.array([[1.,0.,2.]]),np.array([[8.,0.,-1.]]),no_drag,court,dt=.001)
    time=(-1+np.sqrt(1+2*9.81*(2-physics['collision_radius_m'])))/9.81
    assert abs(analytic['landing_m'][0,0]-(1+8*time))<.00001
    output.mkdir(parents=True)
    np.savez_compressed(output/'grid.npz',initial_position_m=initial,outgoing_velocity_mps=outgoing,
        idealized_attack_target_candidates=valid,**result)
    report=dict(created_at=datetime.now(timezone.utc).isoformat(),scope='CPU idealized shooting proposals only; not actual returns, native policy results or whole-robot feasibility proof.',
        source_run=str(run.relative_to(ROOT)),source_env_id=62,source_generation=1,public_clock_s=float(public[0,775]),
        source_actual_contact_clock_s=event['time_s'],source_used_public_observation=public[0].tolist(),states=states,
        numerical_check=dict(coarse_dt_s=.002,fine_dt_s=.001,max_landing_delta_m=float(landing_error.max()),max_margin_delta_m=float(margin_error.max()),constant_acceleration_analytical_check=True),
        caveats=['All computed flight paths omit actual net contact, robot tracking, joint limits and reaction dynamics.',
            'A zero grid count means this finite velocity grid found no proposal, not that the robot task is impossible.',
            'Face normal/normal-speed calculations assume ideal frictionless restitution; native contact must validate them.',
            'Original target, ball, court, attack thresholds and native success definitions are unchanged.'],
        formal_panels_read=False,formal_acceptance=False,
        source_hashes={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__),run/'contacts_001.json',run/'causal_plans.partial.json',ROOT/'configs/shuttle_physics_v2.json',ROOT/'configs/court_net_v4.json']})
    (output/'PLAN.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(states=[{k:v for k,v in state.items() if k not in ['incoming_velocity_mps']} for state in states],numerical_check=report['numerical_check'])),flush=True)


if __name__=='__main__':main()
