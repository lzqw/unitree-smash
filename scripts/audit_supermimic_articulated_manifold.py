#!/usr/bin/env python3
"""Separate articulated recoil from surface geometry using measured impact manifolds."""
import argparse
import json
from pathlib import Path

import torch

from audit_supermimic_paired_body_transfer import response_metrics
from measure_supermimic_native_racket_inertia import digest


def cross_matrix(v):
    x,y,z=v.unbind(-1);o=torch.zeros_like(x)
    return torch.stack((o,-z,y,z,o,-x,-y,x,o),dim=-1).reshape(*v.shape[:-1],3,3)


def coupled_impulse(n, relative, ball_arm, racket_arm, spatial_inverse, *, radius, mass, restitution, mu):
    """Solve the single-substep unilateral contact problem with world-frame point mobilities."""
    count=len(n);eye=torch.eye(3,dtype=n.dtype,device=n.device)
    point=lambda r:torch.cat((eye.expand(count,-1,-1),-cross_matrix(r)),-1).reshape(count*3,6)
    ball=point(ball_arm);racket=point(racket_arm)
    inv_ball=torch.block_diag(eye,eye/(.4*radius**2))
    k=(ball@inv_ball@ball.T+mass*racket@spatial_inverse@racket.T).reshape(count,3,count,3)
    out_speed=(-restitution*(relative*n).sum(-1)).clamp_min(0)
    proj=eye-n[:,:,None]*n[:,None,:]
    dv=torch.zeros_like(n);loads=torch.zeros_like(out_speed);friction=torch.zeros_like(n)
    velocity=relative.clone()
    for _ in range(12):
        for i in range(count):
            normal=n[i];local=k[i,:,i,:]
            update=(out_speed[i]-velocity[i].dot(normal))/normal.dot(local@normal).clamp_min(1.e-12)
            new=(loads[i]+update).clamp_min(0)
            add=normal*(new-loads[i]);loads[i]=new
            velocity+=k[:,:,i,:]@add
            dv[i]+=add
            plane=proj[i]
            solve=plane@local@plane+normal[:,None]*normal[None,:]
            proposed=friction[i]-torch.linalg.solve(solve,plane@velocity[i])
            new_friction=proposed*(mu*new/proposed.norm().clamp_min(1.e-12)).clamp(max=1.)
            add=new_friction-friction[i];friction[i]=new_friction
            velocity+=k[:,:,i,:]@add
            dv[i]+=add
    return (n*loads[:,None]).sum(0),friction.sum(0)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('spec','mobility','calibration','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    meas=json.loads(a.mobility.read_text());cfg=json.loads(a.calibration.read_text())
    phys=json.loads((Path(__file__).resolve().parents[1]/'configs/shuttle_physics_v2.json').read_text())
    variants=('infinite','articulated')
    cases={};stats={}
    for b in ('nominal_a','nominal_b','minus','plus'):
        path=a.spec/b/'native_probe.pt'
        if digest(path)!=meas['sources'][b]:raise ValueError('The contact poses differ from the inertia query')
        src=torch.load(path,weights_only=True,mmap=True);tr=src['raw']['substep_trace'];t=tr['tensors']
        mass,dt=phys['mass_kg'],tr['physics_dt_s']
        for lane in src['spec']['lanes']:
            actual=t['racket_force_w'][1:,lane].double().sum(0)*dt/mass
            tangent=t['racket_friction_force_w'][1:,lane].double().sum(0)*dt/mass
            terms={k:[torch.zeros(3,dtype=torch.float64),torch.zeros(3,dtype=torch.float64)] for k in variants}
            for i,frame in enumerate(tr['manifold_frames'][1:]):
                use=(frame['lane']==lane)&(frame['filter_index']==0)&(frame['normal_force_n'].abs()>.001)
                if not use.any():continue
                props=[k for k in meas['samples'] if (k['branch'],k['lane'],k['frame'])==(b,lane,i+1)]
                if len(props)!=int(use.sum()):raise ValueError('Missing mobility at the native contact tick')
                spatial=torch.tensor(props[0]['spatial_link_mobility'],dtype=torch.float64)
                point=frame['point_w'][use].double()
                n=frame['normal_w'][use].double()*frame['normal_force_n'][use,None].sign()
                incoming=t['shuttle_state_w'][i,lane].double()
                br=point-incoming[:3];rr=point-t['racket_position_w'][i,lane].double()
                rel=(incoming[7:10]+torch.cross(incoming[10:13].expand_as(br),br,dim=-1)
                     -t['racket_linear_velocity_w'][i,lane].double()
                     -torch.cross(t['racket_angular_velocity_w'][i,lane].double().expand_as(rr),rr,dim=-1))
                for model in variants:
                    nr,fr=coupled_impulse(n,rel,br,rr,spatial if model=='articulated' else torch.zeros_like(spatial),
                          radius=phys['collision_radius_m'],mass=mass,restitution=phys['restitution'],mu=cfg['fitted']['friction'])
                    terms[model][0]+=nr;terms[model][1]+=fr
            cases[b,lane]=dict(native=[actual,tangent],**terms)
    lanes=src['spec']['lanes']
    for name,k in [('normal',0),('friction',1),('both',None)]:
        v=lambda b,l,m:sum(cases[b,l][m]) if k is None else cases[b,l][m][k]
        diff=lambda m,x,y:torch.stack([v(x,l,m)-v(y,l,m) for l in lanes])
        stats[name]={m:response_metrics(diff('native','plus','minus'),diff(m,'plus','minus'),diff('native','nominal_a','nominal_b')) for m in variants}
    with a.output.open('x') as out:
        json.dump(dict(physical_mobility_sha256=digest(a.mobility),calibration_sha256=digest(a.calibration),lanes=lanes,
             scope='Native contact-point and normal teacher at loaded frames, causal incoming velocities. No measured load magnitude in the applied impulse. Friction-only frames remain in target.',
             signed=stats,policy_updates=0,contact_bptt_admitted=False),out,indent=2,allow_nan=False)


if __name__=='__main__':main()
