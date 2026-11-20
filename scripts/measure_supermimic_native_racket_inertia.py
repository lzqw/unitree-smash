#!/usr/bin/env python3
"""Query the articulated racket contact-point inverse mass at recorded native impact poses."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file,'sha256').hexdigest()


def measure(args):
    import torch
    from isaaclab.utils.math import quat_apply, quat_mul, quat_conjugate
    from quadsmash.tasks.stand.full_racket_env_v4 import FullRacketEnvV4,full_racket_cfg

    from quadsmash.rl.supermimic_action_identification_dataset_v1 import _same_contract
    captures={}; sources={}
    points=[]
    for branch in ('nominal_a','nominal_b','minus','plus'):
        path=args.spec/branch/'native_probe.pt'
        captures[branch]=digest(path)
        source=torch.load(path,mmap=True,weights_only=True)
        sources[branch]=source
        t=source['raw']['substep_trace']['tensors'];tr=source['raw']['substep_trace']
        selected=set(source['spec']['lanes'])
        for frame,m in enumerate(tr['manifold_frames'][1:],1):
            filtered=(m['filter_index']==0)&(m['normal_force_n'].abs()>.001)
            for idx in filtered.nonzero().flatten().tolist():
                lane=int(m['lane'][idx])
                if lane not in selected: continue
                points.append(dict(branch=branch,lane=lane,frame=frame,
                    q=t['joint_position'][frame-1,lane],qd=t['joint_velocity'][frame-1,lane],
                    racket_q=t['racket_quaternion_w'][frame-1,lane],
                    arm_link=m['point_w'][idx]-t['racket_position_w'][frame-1,lane],
                    n=m['normal_w'][idx]*m['normal_force_n'][idx].sign()))
    _same_contract(sources)
    status=json.loads((args.spec.parent/'status.json').read_text())
    if captures!={s['branch']:s['probe_sha256'] for s in status['completed'] if s['spec']==args.spec.name and s['exit_code']==0}:
        raise ValueError('Source hashes differ from native collection')
    if not points: raise ValueError('No loaded native racket contacts')
    def tensor(name):return torch.stack([x[name] for x in points]).to(args.device)
    cfg=full_racket_cfg(len(points));cfg.sim.device=args.device;cfg.seed=int(source['seed'])
    if digest(Path(cfg.robot.spawn.usd_path))!=source['task_contract']['asset']['sha256']:
        raise ValueError('Native inertia query asset differs from captured task')
    env=FullRacketEnvV4(cfg)
    try:
        r=env.robot;view=r.root_physx_view;rid=env.racket_body_id
        qpos,qvel=tensor('q'),tensor('qd')
        ids=env.joint_ids
        pose=r.data.default_root_state[:,:7].clone()
        pose[:,:3]=env.scene.env_origins
        pose[:,2]+=1
        pose[:,3:]=pose.new_tensor([1.,0.,0.,0.])
        r.write_root_pose_to_sim(pose)
        r.write_joint_state_to_sim(qpos,qvel,joint_ids=ids)
        env.sim.forward()
        def link_quat():
            return view.get_link_transforms()[:,rid,3:7][:,[3,0,1,2]].clone()
        # Reconstruct world orientation with native FK. Translation does not affect inertial response.
        pose[:,3:]=quat_mul(tensor('racket_q'),quat_conjugate(link_quat()))
        r.write_root_pose_to_sim(pose)
        root_twist=pose.new_tensor([.3,-.2,.4,.7,-.4,.9]).repeat(len(points),1)
        r.write_root_velocity_to_sim(root_twist)
        env.sim.forward()
        qr=link_quat()
        angle=2*torch.minimum((qr-tensor('racket_q')).norm(dim=-1),(qr+tensor('racket_q')).norm(dim=-1))
        if angle.max()>.001:raise ValueError(f'Native racket FK mismatch {float(angle.max())} rad')
        torch.testing.assert_close(view.get_dof_positions()[:,ids],qpos,atol=1.e-5,rtol=0)
        jac=view.get_jacobians()[:,rid].double()
        mass=view.get_generalized_mass_matrices().double()
        if jac.shape!=(len(points),6,mass.shape[-1]) or mass.shape[-1]!=qpos.shape[1]+6:
            raise ValueError('Floating articulation mass/Jacobian layout mismatch')
        com_vel=view.get_link_velocities()[:,rid].clone()
        offset=quat_apply(qr,r.data.body_com_pos_b[:,rid])
        link_vel=com_vel.clone();link_vel[:,:3]-=torch.cross(com_vel[:,3:],offset,dim=-1)
        root_link=root_twist.clone()
        root_link[:,:3]-=torch.cross(root_twist[:,3:],quat_apply(pose[:,3:],r.data.body_com_pos_b[:,0]),dim=-1)
        errors={}
        for root_label,root in [('com',root_twist),('link',root_link)]:
            dofs=view.get_dof_velocities()
            result=(jac@torch.cat((root,dofs),dim=1).double()[...,None]).squeeze(-1)
            for site,actual in [('com',com_vel),('link',link_vel)]:
                errors[(root_label,site)]=float((result-actual).abs().max())
        root_label,site=min(errors,key=errors.get)
        if errors[root_label,site]>.001:raise ValueError(f'Unchecked Jacobian convention: {errors}')
        arm=tensor('arm_link').double()-(offset.double() if site=='com' else 0.)
        jpoint=jac[:,:3]+torch.linalg.cross(jac[:,3:].transpose(1,2),arm[:,None,:],dim=-1).transpose(1,2)
        chol=torch.linalg.cholesky(mass)
        link_jac=jac.clone()
        if site=='com':
            link_jac[:,:3] += torch.linalg.cross(jac[:,3:].transpose(1,2),-offset.double()[:,None,:],dim=-1).transpose(1,2)
        spatial=link_jac@torch.cholesky_solve(link_jac.transpose(1,2),chol)
        response=jpoint@torch.cholesky_solve(jpoint.transpose(1,2),chol)
        n=tensor('n').double()
        normal=(n[:,None,:]@response@n[:,:,None]).reshape(-1)
        if (normal<=0).any():raise ValueError('Invalid articulated inverse mass')
        phys=json.loads((Path(__file__).resolve().parents[1]/'configs/shuttle_physics_v2.json').read_text())
        m=phys['mass_kg'];scale=1/(1+m*normal)
        entries=[dict(branch=item['branch'],lane=item['lane'],frame=item['frame'],normal_scale=float(scale[i]),
                      racket_effective_normal_mass_kg=float(1/normal[i]),point_mobility=response[i].tolist(),spatial_link_mobility=spatial[i].tolist()) for i,item in enumerate(points)]
        report=dict(scope='Read-only native M(q) and J(q) at measured pre-impact poses. Free-floating mobility excludes simultaneous court constraints and actuator impulses.',
                    sources=captures,asset_sha256=source['task_contract']['asset']['sha256'],
                    geometry_max_orientation_error_rad=float(angle.max()),
                    jacobian_velocity_error={f'{r}_to_{s}':e for (r,s),e in errors.items()},
                    selected_convention=[root_label,site],samples=entries,shuttle_mass_kg=m,
                    normal_scale_median=float(scale.median()),normal_scale_range=[float(scale.min()),float(scale.max())],
                    actor_updates=0,contact_bptt_admitted=False)
        with args.output.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False)
    finally:env.close()


def main():
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('spec','output','execution-root'):p.add_argument('--'+key,type=Path,required=True)
    AppLauncher.add_app_launcher_args(p)
    args=p.parse_args()
    os.environ['QUADSMASH_G07_EXECUTION_ROOT']=str(args.execution_root.resolve())
    app=AppLauncher(args).app
    try:measure(args)
    finally:app.close()


if __name__=='__main__':main()
