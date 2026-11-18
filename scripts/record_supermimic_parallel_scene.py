#!/usr/bin/env python3
"""Record a true multi-environment Isaac Sim rollout for a parallel-training view."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--execution-root', type=Path, default=Path('/home/autodrive/LZQW/unitree_smash'))
p.add_argument('--num-envs', type=int, default=16)
p.add_argument('--spacing', type=float, default=4.5)
p.add_argument('--seconds', type=float, default=10.0)
p.add_argument('--fps', type=int, default=30)
p.add_argument('--seed', type=int, default=882072)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if args.num_envs < 2: p.error('--num-envs must be >= 2')
if args.spacing <= 0 or args.seconds <= 0 or args.fps <= 0: p.error('spacing, seconds and fps must be positive')
if args.output.exists() and any(args.output.iterdir()): raise ValueError(f'Output directory is not empty: {args.output}')
args.output.mkdir(parents=True, exist_ok=True)

def log(msg): print(f'[PARALLEL] {msg}', flush=True)

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def extract_geometry(env, output):
    from pxr import Usd, UsdGeom, UsdPhysics
    import omni.usd, trimesh, numpy as np
    stage = omni.usd.get_context().get_stage(); cache = UsdGeom.XformCache()
    meshes, mesh_meta = {}, []
    robot_root = stage.GetPrimAtPath('/World/envs/env_0/Robot')
    for prim in Usd.PrimRange(robot_root, Usd.TraverseInstanceProxies()):
        kind = prim.GetTypeName()
        if kind not in ('Mesh','Cube','Sphere','Cylinder','Capsule'): continue
        low = str(prim.GetPath()).lower()
        if '/collisions/' in low or '/collision/' in low: continue
        ancestor = prim
        while ancestor.IsValid() and not ancestor.HasAPI(UsdPhysics.RigidBodyAPI): ancestor = ancestor.GetParent()
        if not ancestor.IsValid() or ancestor.GetName() not in env.robot.body_names: continue
        body = env.robot.body_names.index(ancestor.GetName())
        if kind == 'Mesh':
            shape = UsdGeom.Mesh(prim); vertices = np.asarray(shape.GetPointsAttr().Get(), dtype=np.float64)
            indices = list(shape.GetFaceVertexIndicesAttr().Get()); counts = shape.GetFaceVertexCountsAttr().Get(); faces=[]; off=0
            for count in counts:
                for j in range(1,int(count)-1): faces.append([indices[off],indices[off+j],indices[off+j+1]])
                off += int(count)
            faces = np.asarray(faces, dtype=np.int32)
        else:
            if kind == 'Cube': shape=trimesh.creation.box(extents=[float(UsdGeom.Cube(prim).GetSizeAttr().Get())]*3)
            elif kind == 'Sphere': shape=trimesh.creation.icosphere(subdivisions=2,radius=float(UsdGeom.Sphere(prim).GetRadiusAttr().Get()))
            else:
                us = UsdGeom.Cylinder(prim) if kind=='Cylinder' else UsdGeom.Capsule(prim)
                shape=trimesh.creation.cylinder(radius=float(us.GetRadiusAttr().Get()),height=float(us.GetHeightAttr().Get()),sections=24)
                axis=us.GetAxisAttr().Get()
                if axis=='X': shape.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[0,1,0]))
                elif axis=='Y': shape.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[1,0,0]))
            vertices, faces = np.asarray(shape.vertices,dtype=np.float64), np.asarray(shape.faces,dtype=np.int32)
        matrix=np.asarray(cache.GetLocalToWorldTransform(prim)*cache.GetLocalToWorldTransform(ancestor).GetInverse())
        vertices=(np.c_[vertices,np.ones(len(vertices))]@matrix)[:,:3].astype(np.float32)
        key=f'm{len(mesh_meta)}'; meshes[f'{key}_vertices']=vertices; meshes[f'{key}_faces']=faces
        color=UsdGeom.Gprim(prim).GetDisplayColorAttr().Get()
        mesh_meta.append(dict(key=key,body=body,path=str(prim.GetPath()),color=list(color[0]) if color else None))
    if not mesh_meta: raise RuntimeError('No robot meshes found')
    np.savez_compressed(output/'geometry.npz', **meshes)
    (output/'geometry.json').write_text(json.dumps(dict(meshes=mesh_meta,body_names=env.robot.body_names),indent=2)+'\n')

app = AppLauncher(args).app
try:
    import numpy as np, torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from quadsmash.control.g07_shared_target_residual_v1 import make_g07_residual_controller
    from quadsmash.rl.supermimic_task_contract import make_task_cfg, resolve_task_contract, zero_panel_reset_noise
    from quadsmash.tasks.stand.full_racket_eval_env_v4 import FrozenPanelEnv
    checkpoint=args.checkpoint.resolve(); source=torch.load(checkpoint,map_location='cpu',weights_only=False)
    contract, protocol, cases, receipt=resolve_task_contract(source,checkpoint,execution_root=args.execution_root)
    cfg=make_task_cfg(args.num_envs,contract,protocol,receipt)
    # This is a presentation-only layout. The formal training contract remains
    # origin-spacing 0; visual lanes are separated after contract validation.
    cfg.scene.env_spacing=float(args.spacing)
    cfg.viewer.resolution=(1280,720); cfg.sim.render_interval=max(1,round((1/args.fps)/cfg.sim.dt)); cfg.seed=args.seed; cfg.sim.device='cuda:0'
    log(f'launching one Isaac Sim process with {args.num_envs} parallel envs, spacing={args.spacing}m')
    env=FrozenPanelEnv(cfg,cases=[cases[i%len(cases)] for i in range(args.num_envs)],reset_noise=zero_panel_reset_noise(args.num_envs,args.seed),render_mode=None)
    wrapped=RslRlVecEnvWrapper(env,clip_actions=None); obs,_=wrapped.reset()
    controller=make_g07_residual_controller(source,args.num_envs,env.device,execution_root=args.execution_root); controller.reset()
    default=env.robot.data.default_joint_pos[0,env.joint_ids]; limits=env.robot.data.joint_pos_limits[0,env.joint_ids]
    ticks=int(round(args.seconds/contract['control_dt_s'])); stride=max(1,round(1/(args.fps*contract['control_dt_s'])))
    poses=[]; balls=[]; origins=env.scene.env_origins.detach().cpu().numpy().copy(); t0=time.perf_counter()
    with torch.inference_mode():
        for tick in range(ticks):
            action=controller.action(obs,default,limits); obs,_,_,_=wrapped.step(action)
            if tick%stride==0:
                poses.append(torch.cat((env.robot.data.body_link_pos_w,env.robot.data.body_link_quat_w),dim=2).cpu().numpy().astype('float32').copy())
                balls.append(env.shuttle.data.root_pos_w.cpu().numpy().astype('float32').copy())
            if tick%100==0: log(f'tick={tick}/{ticks} elapsed={time.perf_counter()-t0:.1f}s')
    np.save(args.output/'parallel_link_poses.npy',np.stack(poses)); np.save(args.output/'parallel_ball_positions.npy',np.stack(balls)); np.save(args.output/'env_origins.npy',origins)
    extract_geometry(env,args.output)
    metadata=dict(schema='supermimic-parallel-isaac-scene-v1',checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),num_envs=args.num_envs,spacing_m=args.spacing,seconds=args.seconds,fps=args.fps,frames=len(poses),control_dt_s=contract['control_dt_s'],physics_dt_s=contract['physics_dt_s'],env_origins=origins.tolist(),single_isaac_process=True,source_contract=contract)
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2,default=str)+'\n')
    log(f'capture_complete frames={len(poses)}')
finally:
    try: env.close()
    except Exception: pass
    log('cleanup_done')
    os._exit(0)
