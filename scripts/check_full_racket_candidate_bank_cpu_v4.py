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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--public',type=Path,required=True)
    p.add_argument('--joint-contract',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Use a fresh output directory')
    torch.set_num_threads(2)
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    bank=json.loads(a.bank.read_text());reference=checkpoint['parameters'].numpy()
    low=np.array(LOW+[.8,-.12]+[-5.]*6,dtype=np.float32)
    high=np.array(HIGH+[1.45,.12]+[5.]*6,dtype=np.float32)
    low[22:24]=-.3;high[22:24]=.3
    rows=validate_candidate_bank(bank,checkpoint['seed'],checkpoint['asset_sha256'],checkpoint['parameter_names'],reference,low,high,62)
    assert rows.shape in [(48,32),(60,32)]
    assert np.array_equal(rows[:,26:32],np.broadcast_to(reference[26:32],(len(rows),6)))
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
    paths=[a.checkpoint,a.bank,a.public,a.joint_contract,Path(__file__),ROOT/'quadsmash/control/full_racket_candidate_bank_v4.py']
    report=dict(created_at=datetime.now(timezone.utc).isoformat(),scope='Counterfactual CPU commands on saved old-policy public observations, not changed-state simulation or new policy outcomes.',
                ticks=len(commands),candidates=len(rows),public_case_streams=2,lanes=count,
                observation_width=794,action_width=18,all_actions_finite_and_bounded=True,
                all_proposals_inside_original_search_bounds=True,all_velocity_directions_equal_measured_power_reference=True,
                planning_completed_for_all_lanes=True,normalized_action_saturation_fraction=float(np.mean(np.abs(commands)>=.99999)),
                native_execution=False,formal_acceptance=False,
                sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
    (a.output/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
