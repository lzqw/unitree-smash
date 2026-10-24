#!/usr/bin/env python3
"""Replay saved public inputs through legacy and optional new control paths.

This checks command compatibility and interfaces, not new native dynamics.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import types

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from quadsmash.control.full_racket_rally_controller_v4 import FullRacketRallyControllerV4
from quadsmash.control.full_racket_arm_search_v4 import extend_arm_velocity_parameters


def load_legacy(path,name):
    module=types.ModuleType(name)
    module.__file__=str(path)
    module.__package__='quadsmash.control'
    exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)
    return module


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy-root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--public',type=Path,required=True)
    p.add_argument('--joint-contract',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Use a fresh output directory')
    torch.set_num_threads(2)
    legacy_teacher_path=a.legacy_root/'quadsmash/control/full_racket_teacher_v4.py'
    legacy_controller_path=a.legacy_root/'quadsmash/control/full_racket_rally_controller_v4.py'
    teacher=load_legacy(legacy_teacher_path,'legacy_teacher')
    legacy=load_legacy(legacy_controller_path,'legacy_rally')
    legacy.FullRacketTeacherV4=teacher.FullRacketTeacherV4
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    assert len(checkpoint['parameters'])==26
    augmented=copy.deepcopy(checkpoint)
    augmented['parameters']=extend_arm_velocity_parameters(checkpoint['parameters'],checkpoint['seed']['joint_velocity_seed_rad_s'])
    adaptive=copy.deepcopy(augmented)
    adaptive['seed']=dict(adaptive['seed'],causal_arm_prepose_gain=.5)
    controllers=[legacy.FullRacketRallyControllerV4(checkpoint,2,'cpu'),
                 FullRacketRallyControllerV4(checkpoint,2,'cpu'),
                 FullRacketRallyControllerV4(augmented,2,'cpu'),
                 FullRacketRallyControllerV4(adaptive,2,'cpu')]
    contract=json.loads(a.joint_contract.read_text())
    names=[f'{leg}_{joint}_joint' for joint in ['hip','thigh','calf'] for leg in ['FL','FR','RL','RR']]+[f'joint{i}' for i in range(1,7)]
    defaults=contract['defaults_before_neutral_arm_override']
    reset=json.loads((ROOT/'configs/full_racket_reset_v4.json').read_text())['arm_joint_positions_rad']
    defaults.update({f'joint{i+1}':q for i,q in enumerate(reset)})
    default=torch.tensor([defaults[n] for n in names],dtype=torch.float32)
    limits=torch.tensor([contract['limits'][n] for n in names],dtype=torch.float32)
    data=np.load(a.public)
    recorded=[[] for _ in controllers]
    for controller in controllers:controller.reset()
    with torch.inference_mode():
        for index,obs in enumerate(data['observations']):
            if not data['valid_first_episode'][index].all():break
            observation={'policy':torch.tensor(obs)}
            for controller,output in zip(controllers,recorded):
                action=controller.action(observation,default,limits)
                assert action.shape==(2,18) and torch.isfinite(action).all() and (action.abs()<=1).all()
                output.append(action.numpy().copy())
    values=[np.stack(x) for x in recorded]
    assert len(values[0])>200
    assert np.array_equal(values[0],values[1])
    assert np.array_equal(values[1],values[2])
    assert not np.array_equal(values[2],values[3])
    assert all(controller.planned.all() for controller in controllers)
    plan=controllers[3].logs[0]
    delta=np.asarray(plan['causal_prepose_joint_delta_rad'])
    assert np.isfinite(delta).all() and np.any(delta[0]!=delta[1])
    assert max(plan['observation_clock_s'])<2.1
    assert (controllers[3].parameters[:,8:14].abs()<=.25).all()
    planned_hit=controllers[3].teacher.arm[None]+controllers[3].parameters[:,8:14]
    assert (planned_hit>=limits[12:,0]-1e-6).all() and (planned_hit<=limits[12:,1]+1e-6).all()
    assert np.abs(delta).max()<=.300001
    original_hit=(controllers[2].teacher.arm[None]+controllers[2].parameters[:,8:14]).clamp(limits[12:,0],limits[12:,1])
    kin=controllers[3].teacher.kinematics
    with torch.inference_mode():
        old_point,old_normal,_,_=kin(original_hit,controllers[3].teacher.contact_offset)
        new_point,new_normal,_,_=kin(planned_hit,controllers[3].teacher.contact_offset)
    public=np.asarray(plan['used_public_observations'])
    forward,up=public[:,763:766],-public[:,6:9]
    world_delta=np.asarray(plan['position_m'])-controllers[3].nominal_target
    body_delta=.5*(world_delta[:,:1]*forward+world_delta[:,1:2]*np.cross(up,forward)+world_delta[:,2:3]*up)
    position_error=np.linalg.norm(new_point.numpy()-old_point.numpy()-body_delta,axis=1)
    unchanged_error=np.linalg.norm(body_delta,axis=1)
    assert (position_error<unchanged_error).all()
    normal_degrees=np.degrees(np.arccos(np.clip((old_normal.numpy()*new_normal.numpy()).sum(1),-1,1)))
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'counterfactual_commands.npz',legacy26=values[0],updated26=values[1],
                        updated32=values[2],updated32_prepose=values[3])
    paths=[a.checkpoint,a.public,a.joint_contract,legacy_teacher_path,legacy_controller_path,
           ROOT/'quadsmash/control/full_racket_arm_search_v4.py',ROOT/'quadsmash/control/full_racket_teacher_v4.py',
           ROOT/'quadsmash/control/full_racket_rally_controller_v4.py',Path(__file__)]
    report=dict(created_at=datetime.now(timezone.utc).isoformat(),
                scope='CPU command replay on saved old-policy public observations. New dynamics, contact and recovery are not executed.',
                ticks=len(values[0]),lanes=2,observation_width=794,action_width=18,
                legacy26_bitwise_equal=True,embedded32_bitwise_equal_to26=True,
                all_actions_finite_and_bounded=True,adaptive_prepose_has_distinct_causal_lane_offsets=True,
                planning_clock_s=plan['observation_clock_s'],applied_joint_delta_rad=delta.tolist(),
                all_adaptive_pose_offsets_in_original_search_bounds=True,native_execution=False,formal_acceptance=False,
                planned_hit_joints_within_original_hardware_position_limits=True,
                static_planned_position_errors_m=position_error.tolist(),unchanged_static_position_errors_m=unchanged_error.tolist(),
                static_planned_face_normal_changes_deg=normal_degrees.tolist(),
                sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
    (a.output/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
