#!/usr/bin/env python3
"""Check native racket friction anchors against observed ball torque and tangential motion."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from audit_supermimic_pilot_tangential_transfer import sha


def sticking_delta(arms, slips, normal, inertia_per_mass, max_dv):
    """Solve simultaneous tangential sticking constraints for one sphere and a moving surface."""
    # The contact frame is used only for an offline linear solve.
    axis = torch.eye(3,dtype=normal.dtype)[normal.abs().argmin()]
    u = F.normalize(torch.cross(normal, axis,dim=-1),dim=-1)
    tangents = torch.stack((u,torch.cross(normal,u,dim=-1)))
    # Columns of the impulse Jacobian map tangent impulses to linear/angular velocity.
    lin = tangents.repeat(len(arms),1)
    arm = arms.repeat_interleave(2,dim=0)
    angular = torch.cross(arm,lin,dim=-1)
    effective = lin@lin.T + (angular@angular.T)/inertia_per_mass
    projected = (slips@tangents.T).flatten()
    impulse = -torch.linalg.pinv(effective,hermitian=True,rtol=1e-7)@projected
    result = impulse@lin
    return result*(max_dv/result.norm().clamp_min(1e-12)).clamp(max=1.)


def inspect(trace, physics, mu):
    frames=trace['racket_friction_frames']
    manifold=trace['manifold_frames']
    t=trace['tensors']
    if len(frames)!=trace['count'] or len(manifold)!=len(frames):
        raise ValueError('Missing native friction point frames')
    dt=trace['physics_dt_s']; mass=physics['mass_kg']; radius=physics['collision_radius_m']
    if dt!=physics['native_physics_dt_s']:
        raise ValueError('Different physical timebase')
    inertia=.4*radius**2
    rows=[]
    for i in range(1,len(frames)):
        fr=frames[i]; nf=manifold[i]
        candidate=(t['racket_friction_force_w'][i].norm(dim=-1)>.001)|(t['racket_force_w'][i].norm(dim=-1)>.001)
        isolated=(t['ground_force_w'][i].norm(dim=-1)<=.001)&(t['net_force_matrix_w'][i].reshape(len(candidate),-1,3).norm(dim=-1).amax(-1)<=.001)
        for lane in (candidate&isolated).nonzero().flatten().tolist():
            incoming=t['shuttle_state_w'][i-1,lane].double()
            center=incoming[:3]
            selection=fr['lane']==lane
            arms=fr['point_w_m'][selection].double()-center
            friction=fr['force_w_n'][selection].double()*dt/mass
            use=(nf['lane']==lane)&(nf['filter_index']==0)
            jn=nf['normal_force_n'][use,None].double()*nf['normal_w'][use].double()*dt/mass
            rn=nf['point_w'][use].double()-center
            normal=jn.sum(0)
            tangent=friction.sum(0)
            torch.testing.assert_close(normal,t['racket_force_w'][i,lane].double()*dt/mass,rtol=1e-5,atol=1e-5)
            torch.testing.assert_close(tangent,t['racket_friction_force_w'][i,lane].double()*dt/mass,rtol=1e-5,atol=1e-5)
            spin_n=torch.cross(rn,jn,dim=-1).sum(0)/inertia
            spin_f=torch.cross(arms,friction,dim=-1).sum(0)/inertia
            spin=t['shuttle_state_w'][i,lane,10:13].double()-incoming[10:13]
            n=F.normalize(normal,dim=-1)
            racket=t['racket_position_w'][i-1,lane].double()
            linear=t['racket_linear_velocity_w'][i-1,lane].double()
            angular=t['racket_angular_velocity_w'][i-1,lane].double()
            def predict(lever,delta_spin):
                if not len(lever) or normal.norm()==0:
                    return torch.zeros_like(normal)
                w=(incoming[10:13]+delta_spin).expand_as(lever)
                rel=incoming[7:10]+normal+torch.cross(w,lever,dim=-1)-linear-torch.cross(angular.expand_as(lever),center+lever-racket,dim=-1)
                return sticking_delta(lever,rel,n,inertia,mu*normal.norm())
            real=predict(arms,spin_n)
            radial=predict((-radius*n)[None],torch.zeros_like(spin_n))
            carrying = friction.norm(dim=-1)>.001*dt/mass
            carrying_oracle = predict(arms[carrying],spin_n)
            # A single effective anchor keeps the free torsional degree of freedom.
            # Multiple nearly coincident sticking constraints can otherwise demand
            # a large cancelling pair of impulses before Coulomb projection.
            centroid = arms.mean(0,keepdim=True) if len(arms) else arms
            one_anchor = predict(centroid,spin_n)
            normal_weight = jn.norm(dim=-1)
            weighted_point = (normal_weight[:,None]*rn).sum(0)/normal_weight.sum().clamp_min(1e-12)
            closest = arms[(arms-weighted_point).norm(dim=-1).argmin()][None] if len(arms) else arms
            nearest_loaded = predict(closest,spin_n)
            radial_spin=torch.cross(-radius*n,tangent,dim=-1)/inertia
            rows.append(dict(tick=i,lane=lane,normal_points=int((jn.norm(dim=-1)>dt*.001/mass).sum()),
                friction_anchors=len(arms),radial_surface_error_m=(arms.norm(dim=-1)-radius).abs().tolist(),
                active_friction_anchors=int(carrying.sum()),native=tangent.tolist(),anchor_teacher=real.tolist(),
                radial_teacher=radial.tolist(),centroid_teacher=one_anchor.tolist(),
                nearest_loaded_normal_teacher=nearest_loaded.tolist(),friction_support_oracle=carrying_oracle.tolist(),
                native_spin_rad_s=spin.tolist(),torque_spin_rad_s=(spin_n+spin_f).tolist(),
                radial_spin_rad_s=radial_spin.tolist()))
    return rows


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--collection',type=Path,required=True)
    cli.add_argument('--calibration',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args();torch.set_num_threads(2)
    prov=json.loads((args.collection/'provenance.json').read_text())
    phys=json.loads((Path(__file__).resolve().parents[1]/'configs/shuttle_physics_v2.json').read_text())
    mu=json.loads(args.calibration.read_text())['fitted']['friction']
    data=[];archives={}
    for seed in range(prov['seed'],prov['seed']+prov['batches']):
        path=args.collection/f'spec_{seed}'/'trace.pt'
        archives[str(seed)]=sha(path)
        trial=torch.load(path,weights_only=True,map_location='cpu',mmap=True)
        if trial['seed']!=seed:raise ValueError('Pilot seed differs')
        data.extend(dict(seed=seed,**r) for r in inspect(trial['raw']['substep_trace'],phys,mu))
    if not data:raise ValueError('No measured racket response')
    stack=lambda k:torch.tensor([r[k] for r in data],dtype=torch.float64)
    def rmse(x,y):return float((stack(x)-stack(y)).square().mean().sqrt())
    metrics={key:rmse(key,'native') for key in ('anchor_teacher','radial_teacher','centroid_teacher','nearest_loaded_normal_teacher','friction_support_oracle')}
    spin_metrics={key:rmse(key,'native_spin_rad_s') for key in ('torque_spin_rad_s','radial_spin_rad_s')}
    out=dict(archives=archives,provenance_sha256=sha(args.collection/'provenance.json'),calibration_sha256=sha(args.calibration),
             trials=prov['lanes']*prov['batches'],loaded_substeps=len(data),friction_component_rmse_mps=metrics,
             angular_component_rmse_rad_s=spin_metrics,samples=data,policy_training_admitted=False,
             scope='Isolated measured forces; teacher uses native anchor locations, normal torque and incoming states. Absolute errors retain frames with missing normal or friction.')
    with args.output.open('x') as f:json.dump(out,f,indent=2,allow_nan=False)


if __name__=='__main__':
    main()
