#!/usr/bin/env python3
"""Render a preselected ground scenario with the frozen autonomous G04 actor."""
import argparse,hashlib,json,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--checkpoint',type=Path,default=ROOT/'artifacts/G04/policy.pt')
p.add_argument('--trajectory-only',action='store_true',help='Record actual link poses and geometry for offline rendering without RTX')
AppLauncher.add_app_launcher_args(p);args=p.parse_args()
app=AppLauncher(args).app
import numpy as np
import torch
import imageio.v2 as imageio
import isaaclab.sim as sim_utils
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.tasks.stand.jump_env import JumpEnv,JumpEnvCfg,PROTOCOL
from quadsmash.rl.native_ppo import NativeRunner
env=None;writer=None
try:
    cfg=JumpEnvCfg();cfg.scene.num_envs=1;cfg.seed=1907;cfg.sim.device='cuda:0'
    cfg.viewer.eye=(1.75,-2.35,1.05);cfg.viewer.lookat=(0.,0.,.48)
    cfg.viewer.resolution=(1280,720);cfg.sim.render_interval=10
    env=JumpEnv(cfg,render_mode=None if args.trajectory_only else 'rgb_array');wrapped=RslRlVecEnvWrapper(env,clip_actions=None)
    if not args.trajectory_only:
        light=sim_utils.DomeLightCfg(intensity=2200.,color=(.92,.95,1.))
        light.func('/World/VideoLight',light)
    config=json.loads((ROOT/'configs/g04_imitation.json').read_text())
    runner=NativeRunner(wrapped,config['runner'],log_dir=str(args.output/'policy'),device=env.device)
    runner.load(str(args.checkpoint),load_optimizer=False)
    policy=runner.get_inference_policy(device=env.device)
    panel=PROTOCOL['panels'][1];rng=np.random.Generator(np.random.PCG64(panel['seed']));count=panel['episodes']
    dq=rng.uniform(-panel['joint_offset_uniform_rad'],panel['joint_offset_uniform_rad'],(count,18))
    qd=rng.uniform(-panel['joint_velocity_uniform_rad_s'],panel['joint_velocity_uniform_rad_s'],(count,18))
    yaw=rng.uniform(-panel['yaw_uniform_rad'],panel['yaw_uniform_rad'],count)
    xy=rng.uniform(-panel['xy_uniform_m'],panel['xy_uniform_m'],(count,2))
    with torch.inference_mode():
        env._episode_has_stepped.zero_();env._metrics_started=False;wrapped.reset();env.episode_records.clear()
        root=env.robot.data.root_state_w.clone();root[:,:2]+=torch.tensor(xy[:1],device=env.device,dtype=torch.float32)
        angle=torch.tensor(yaw[:1],device=env.device,dtype=torch.float32)
        root[:,3]=torch.cos(angle/2);root[:,4:6]=0;root[:,6]=torch.sin(angle/2);root[:,7:]=0
        q=env.robot.data.default_joint_pos.clone();v=torch.zeros_like(q)
        q[:,env.joint_ids]+=torch.tensor(dq[:1],device=env.device,dtype=torch.float32)
        v[:,env.joint_ids]=torch.tensor(qd[:1],device=env.device,dtype=torch.float32)
        env.robot.write_root_state_to_sim(root);env.robot.write_joint_state_to_sim(q,v);env.sim.forward()
        env.policy_history.reset(torch.arange(1,device=env.device));obs=wrapped.get_observations()
        # Warm only the renderer; physics and the policy clock do not advance.
        if args.trajectory_only:
            from pxr import Usd,UsdGeom,UsdPhysics
            import omni.usd
            import trimesh
            stage=omni.usd.get_context().get_stage();cache=UsdGeom.XformCache()
            meshes={};meta=[]
            for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/envs/env_0/Robot'),Usd.TraverseInstanceProxies()):
                kind=prim.GetTypeName()
                if kind not in ('Mesh','Cube','Sphere','Cylinder','Capsule'):continue
                if '/collisions/' in str(prim.GetPath()).lower() or '/collision/' in str(prim.GetPath()).lower():continue
                ancestor=prim
                while ancestor.IsValid() and not ancestor.HasAPI(UsdPhysics.RigidBodyAPI):ancestor=ancestor.GetParent()
                if not ancestor.IsValid() or ancestor.GetName() not in env.robot.body_names:continue
                body=env.robot.body_names.index(ancestor.GetName())
                if kind=='Mesh':
                    shape=UsdGeom.Mesh(prim);verts=np.asarray(shape.GetPointsAttr().Get(),dtype=np.float64)
                    indices=list(shape.GetFaceVertexIndicesAttr().Get());counts=shape.GetFaceVertexCountsAttr().Get();faces=[];offset=0
                    for size in counts:
                        for j in range(1,size-1):faces.append([indices[offset],indices[offset+j],indices[offset+j+1]])
                        offset+=size
                    faces=np.asarray(faces,dtype=np.int32)
                else:
                    if kind=='Cube':shape=trimesh.creation.box(extents=[float(UsdGeom.Cube(prim).GetSizeAttr().Get())]*3)
                    elif kind=='Sphere':shape=trimesh.creation.icosphere(subdivisions=2,radius=float(UsdGeom.Sphere(prim).GetRadiusAttr().Get()))
                    else:
                        usdshape=UsdGeom.Cylinder(prim) if kind=='Cylinder' else UsdGeom.Capsule(prim)
                        shape=trimesh.creation.cylinder(radius=float(usdshape.GetRadiusAttr().Get()),height=float(usdshape.GetHeightAttr().Get()),sections=32)
                        axis=usdshape.GetAxisAttr().Get()
                        if axis=='X':shape.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[0,1,0]))
                        if axis=='Y':shape.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[1,0,0]))
                    verts=np.asarray(shape.vertices);faces=np.asarray(shape.faces,dtype=np.int32)
                matrix=np.asarray(cache.GetLocalToWorldTransform(prim)*cache.GetLocalToWorldTransform(ancestor).GetInverse())
                vertices=(np.c_[verts,np.ones(len(verts))]@matrix)[:,:3].astype(np.float32)
                key=f'm{len(meta)}';meshes[key+'_vertices']=vertices;meshes[key+'_faces']=faces
                color=UsdGeom.Gprim(prim).GetDisplayColorAttr().Get()
                meta.append(dict(key=key,body=body,path=str(prim.GetPath()),color=list(color[0]) if color else None))
            if not meta:raise RuntimeError('no body-local render meshes found')
            np.savez_compressed(args.output/'geometry.npz',**meshes)
            (args.output/'geometry.json').write_text(json.dumps(dict(meshes=meta,body_names=env.robot.body_names),indent=2))
            poses=[]
        else:
            for _ in range(12):env.render()
            writer=imageio.get_writer(str(args.output/'raw_50fps.mp4'),fps=50,codec='libx264',quality=8,macro_block_size=8)
        trace=[]
        for tick in range(400):
            if tick%2==0:
                if args.trajectory_only:
                    poses.append(torch.cat((env.robot.data.body_link_pos_w[0],env.robot.data.body_link_quat_w[0]),1).cpu().numpy().copy())
                else:
                    frame=env.render()
                    if not frame.any():raise RuntimeError('empty RGB rendering')
                    writer.append_data(frame)
                    if tick in (0,120,140,180,300):imageio.imwrite(args.output/f'frame_{tick:03d}.png',frame)
                trace.append(dict(time_s=tick*.01,com_rise_m=float(env.max_com_rise[0]),
                    airborne_s=float(env.airborne_time[0]),recovery_s=float(env.recovery_time[0]),
                    support_feet=int(env.observed_foot_contact[0].sum()),body_contact=bool(env.body_ground_seen[0])))
            obs,_,_,_=wrapped.step(policy(obs))
        if writer is not None:writer.close();writer=None
        if args.trajectory_only:np.save(args.output/'link_poses.npy',np.stack(poses))
    assert len(env.episode_records)==1
    report=dict(kind='rendered demonstration, not a new 400-episode acceptance evaluation',
        panel=panel['id'],scenario=0,selection='scenario chosen before outcomes',
        checkpoint=str(args.checkpoint),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        episode=env.episode_records[0],frames=200,fps=50,env_steps=400,
        assistance=False,policy='deterministic neural actor from ground',
        visual_changes_only=['camera','dome light','render interval'])
    report['capture_mode']='actual body poses for offline rendering' if args.trajectory_only else 'native RTX'
    (args.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output/'trace.json').write_text(json.dumps(trace,indent=2)+'\n')
    print(json.dumps(report),flush=True)
except BaseException:
    (args.output/'error.txt').write_text(traceback.format_exc());raise
finally:
    if writer is not None:writer.close()
    if env is not None:env.close()
    app.close(skip_cleanup=True)
