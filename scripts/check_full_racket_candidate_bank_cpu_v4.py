#!/usr/bin/env python3
"""Check bounded candidate commands on frozen public inputs, without dynamics."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from quadsmash.control.full_racket_rally_controller_v4 import FullRacketRallyControllerV4
from quadsmash.control.full_racket_candidate_bank_v4 import validate_candidate_bank
from quadsmash.control.full_racket_teacher_v4 import LOW,HIGH
from quadsmash.control.full_racket_contact_curriculum_v4 import validate_recovery_search_min
from quadsmash.control.full_racket_recovery_search_v4 import RECOVERY_LOW,RECOVERY_HIGH
from quadsmash.control.full_racket_parameter_layout_v4 import controller_parameter_names
from quadsmash.control.full_racket_contact_normal_search_v4 import NORMAL_LOW,NORMAL_HIGH


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--public',type=Path,required=True)
    p.add_argument('--joint-contract',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-searched-velocities',action='store_true')
    p.add_argument('--recovery-search-min',type=float,default=1.55)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Use a fresh output directory')
    torch.set_num_threads(2)
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    bank=json.loads(a.bank.read_text());reference=checkpoint['parameters'].numpy()
    parameter_names=controller_parameter_names(search_arm_velocity=True,search_recovery_parameters=len(reference)>=37,
        search_contact_normal=len(reference)==38)
    assert checkpoint['parameter_names']==parameter_names, 'Checkpoint differs from native runtime parameter layout'
    low=np.array(LOW+[.8,-.12]+[-5.]*6,dtype=np.float32)
    high=np.array(HIGH+[1.45,.12]+[5.]*6,dtype=np.float32)
    low[22:24]=-.3;high[22:24]=.3
    low[3]=validate_recovery_search_min(a.recovery_search_min)
    if len(reference)>=37:
        low=np.r_[low,np.array(RECOVERY_LOW,dtype=np.float32)]
        high=np.r_[high,np.array(RECOVERY_HIGH,dtype=np.float32)]
    if len(reference)==38:
        low=np.r_[low,np.array(NORMAL_LOW,dtype=np.float32)]
        high=np.r_[high,np.array(NORMAL_HIGH,dtype=np.float32)]
    rows=validate_candidate_bank(bank,checkpoint['seed'],checkpoint['asset_sha256'],parameter_names,reference,low,high,62)
    assert rows.shape in [(48,32),(60,32),(60,37),(61,37),(60,38),(61,38)]
    preserved=np.array_equal(rows[:,26:32],np.broadcast_to(reference[26:32],(len(rows),6)))
    if not a.allow_searched_velocities:assert preserved
    else:assert (np.abs(rows[:,26:32])<=5).all() and (np.abs(rows[:,26:32]*rows[:,7:8])<=5).all()
    count=2*len(rows)
    controller=FullRacketRallyControllerV4(checkpoint,count,'cpu');controller.reset()
    controller.parameters.copy_(torch.tensor(np.repeat(rows,2,axis=0)))
    contract=json.loads(a.joint_contract.read_text())
    names=[f'{leg}_{joint}_joint' for joint in ['hip','thigh','calf'] for leg in ['FL','FR','RL','RR']]+[f'joint{i}' for i in range(1,7)]
    defaults=contract['defaults_before_neutral_arm_override']
    reset=json.loads((ROOT/'configs/full_racket_reset_v4.json').read_text())['arm_joint_positions_rad']
    defaults.update({f'joint{i+1}':q for i,q in enumerate(reset)})
    default=torch.tensor([defaults[n] for n in names],dtype=torch.float32)
    limits=torch.tensor([contract['limits'][n] for n in names],dtype=torch.float32)
    values=[]
    with np.load(a.public) as recorded,torch.inference_mode():
        for index,obs in enumerate(recorded['observations']):
            if not recorded['valid_first_episode'][index].all():break
            public=torch.tensor(np.tile(obs,(len(rows),1)))
            action=controller.action({'policy':public},default,limits)
            assert action.shape==(count,18) and torch.isfinite(action).all() and (action.abs()<=1).all()
            values.append(action.numpy().copy())
    commands=np.stack(values);assert len(commands)>200 and controller.planned.all()
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'counterfactual_commands.npz',actions=commands)
    paths=[a.checkpoint,a.bank,a.public,a.joint_contract,Path(__file__)]
    paths += [ROOT/'quadsmash/control'/name for name in ['full_racket_candidate_bank_v4.py','full_racket_contact_curriculum_v4.py','full_racket_recovery_search_v4.py','full_racket_arm_search_v4.py','full_racket_parameter_layout_v4.py',
        'full_racket_teacher_v4.py','full_racket_kinematics_v4.py','full_racket_joint_preparation_v4.py','full_racket_contact_normal_search_v4.py',
        'full_racket_rally_controller_v4.py','full_racket_relative_prepose_v4.py']]
    report=dict(created_at=datetime.now(timezone.utc).isoformat(),scope='Counterfactual CPU commands on saved old-policy public observations, not changed-state simulation or new policy outcomes.',
                ticks=len(commands),candidates=len(rows),public_case_streams=2,lanes=count,
                parameter_width=len(reference),standing_recovery_parameters_searched=bool(len(reference)>=37),
                contact_normal_bias_searched=bool(len(reference)==38),
                observation_width=794,action_width=18,all_actions_finite_and_bounded=True,
                all_proposals_inside_original_search_bounds=(bool((rows[:,3]>=LOW[3]).all()) if len(reference)==32 else None),
                original_bounds_scope='Original32-coordinate search; added37/38policy dimensions have separately declared algorithm bounds.',
                all_handover_values_inside_original_bounds=bool((rows[:,3]>=LOW[3]).all()),
                all_proposals_inside_declared_algorithm_search_bounds=True,
                recovery_search_min_local_s=a.recovery_search_min,
                physical_hardware_limits_and_success_gates_unchanged=True,
                all_velocity_directions_equal_measured_power_reference=bool(preserved),
                searched_velocity_seeds_explicitly_enabled=bool(a.allow_searched_velocities),
                planning_completed_for_all_lanes=True,normalized_action_saturation_fraction=float(np.mean(np.abs(commands)>=.99999)),
                native_execution=False,formal_acceptance=False,
                sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
    (a.output/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
