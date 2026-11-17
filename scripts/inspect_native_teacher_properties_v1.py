#!/usr/bin/env python3
"""Read realized PhysX properties after reset; zero control steps or policy calls."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--precision-receipt',type=Path,required=True)
p.add_argument('--historical-observations',type=Path,required=True)
AppLauncher.add_app_launcher_args(p);args=p.parse_args()
if args.output.exists():raise ValueError('Fresh inspection directory required')
args.output.mkdir(parents=True)
app=AppLauncher(args).app
import numpy as np
import omni.usd
from pxr import PhysxSchema
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.tasks.stand.contact_recovery_origin0_v5 import contact_recovery_origin0_cfg
from quadsmash.tasks.stand.full_racket_eval_env_v4 import FrozenPanelEnv
from quadsmash.control.full_racket_train_evaluation_v4 import training_noise


def main():
    protocol=json.loads((ROOT/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json').read_text())
    suite=ROOT/protocol['training_suite']['path'];asset=ROOT/protocol['asset']['path']
    for file,digest in [(suite,protocol['training_suite']['sha256']),(asset,protocol['asset']['sha256'])]:
        if hashlib.sha256(file.read_bytes()).hexdigest()!=digest:raise ValueError('Frozen input drift')
    cases=json.loads(suite.read_text())['cases'];cases=[cases[0],cases[64]]
    cfg=contact_recovery_origin0_cfg(2,protocol,args.precision_receipt);cfg.seed=882072;cfg.ground_size_m=(64.,64.,.1)
    noise=training_noise(2,dict(joint_position_rad=0.,joint_velocity_rad_s=0.,root_xy_m=0.,root_yaw_rad=0.),cfg.seed)
    env=FrozenPanelEnv(cfg,cases=cases,reset_noise=noise,contact_flight_recovery=protocol['recovery_association']['schema'])
    try:
        obs,_=RslRlVecEnvWrapper(env,clip_actions=None).reset()
        historical=np.load(args.historical_observations,mmap_mode='r',allow_pickle=False)
        delta=float(np.max(abs(obs['policy'].detach().cpu().numpy()-historical[0])))
        if delta>1e-4:raise ValueError('Historical initial state differs')
        view=env.robot.root_physx_view;values={}
        for name in ('masses','inertias','coms','dof_armatures','dof_stiffnesses','dof_dampings',
                     'dof_max_forces','dof_max_velocities','dof_limits'):
            value=getattr(view,'get_'+name)().detach().cpu().numpy()
            if not np.isfinite(value).all():raise ValueError('Nonfinite realized '+name)
            values[name]=value
        friction_name='dof_friction_coefficients' if hasattr(view,'get_dof_friction_coefficients') else 'dof_friction_properties'
        values[friction_name]=getattr(view,'get_'+friction_name)().detach().cpu().numpy()
        np.savez_compressed(args.output/'PROPERTIES.npz',**values)
        stage=omni.usd.get_context().get_stage();bodies=[]
        for path in view.link_paths[0]:
            prim=stage.GetPrimAtPath(path);api=PhysxSchema.PhysxRigidBodyAPI(prim)
            bodies.append(dict(path=path,physx_api_applied=prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI),
                angular_damping=api.GetAngularDampingAttr().Get(),linear_damping=api.GetLinearDampingAttr().Get(),
                gyroscopic_forces=api.GetEnableGyroscopicForcesAttr().Get()))
        record=dict(body_names=list(env.robot.body_names),joint_names=list(env.robot.joint_names),
            link_paths=view.link_paths,dof_paths=view.dof_paths,bodies=bodies,
            inertia_semantics='column-major 3x3 at COM in rigid-body-prim frame; get_coms quaternion xyzw',
            initial_observation_delta=delta,episode_lengths=env.episode_length_buf.detach().cpu().tolist(),
            control_steps_called=0,actor_forward_calls=0,optimizer_steps=0,policy_evaluation=False,
            asset_sha256=protocol['asset']['sha256'],physics_dt_s=cfg.sim.dt,
            properties_sha256=hashlib.sha256((args.output/'PROPERTIES.npz').read_bytes()).hexdigest(),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (args.output/'PROPERTIES.json').write_text(json.dumps(record,indent=2))
        if any(record['episode_lengths']):raise ValueError('Unexpected control advancement')
        (args.output/'PROPERTY_INSPECTION_COMPLETE.json').write_text(json.dumps(dict(control_steps=0,
            native_policy_evaluation=False,physics_trajectory_calibration=False,initial_state_matches=True),indent=2))
    finally:
        env.close()


try:
    main()
finally:
    app.close()
