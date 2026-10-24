#!/usr/bin/env python3
"""Prepare a fresh task-space preparation family; inherit no policy fitness."""
import argparse
import copy
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from quadsmash.control.full_racket_preparation_v4 import validate_preparation
from quadsmash.control.full_racket_candidate_bank_v4 import validate_candidate_bank
from quadsmash.control.full_racket_teacher_v4 import LOW,HIGH


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Use a fresh output directory')
    source=torch.load(a.checkpoint,map_location='cpu',weights_only=True);bank=json.loads(a.bank.read_text())
    assert bank['seed']==source['seed'] and len(source['parameters'])==32
    config=validate_preparation(dict(position_world_m=[1.,0.,1.15],transition_duration_s=.55,
        servo_gain=1.,nullspace_gain=.05,unhit_reference_max_x_m=1.40))
    warm=copy.deepcopy(source);warm['seed']=dict(source['seed'],cartesian_preparation=config)
    warm['parameters']=warm['parameters'].clone();old_gain=float(warm['parameters'][7]);warm['parameters'][7]=.55
    warm.pop('training_result',None)
    warm.update(kind='Unexecuted Cartesian preparation family; actual outcomes must be remeasured from ground.',
                native_execution=False,performance_inherited=False,formal_acceptance=False)
    rows=np.asarray(bank['parameters'],dtype=np.float32)
    rows[:,7]=np.where(np.abs(rows[:,7]-old_gain)<1e-6,.55,.75)
    bank=copy.deepcopy(bank);bank.update(seed=warm['seed'],reference_parameters=warm['parameters'].tolist(),
        parameters=rows.tolist(),controller_revision='causal-cartesian-preparation-v1',native_execution=False,
        scope='Same body proposals with a new public task-space arm controller; no native power/contact/recovery inherited.')
    for row,label in zip(rows,bank['proposal_labels']):
        if 'arm_velocity_gain' in label:label['arm_velocity_gain']=float(row[7])
        label['task_space_velocity_gain']=float(row[7])
    low=np.array(LOW+[.8,-.12]+[-5.]*6,dtype=np.float32);high=np.array(HIGH+[1.45,.12]+[5.]*6,dtype=np.float32)
    low[22:24]=-.3;high[22:24]=.3
    checked=validate_candidate_bank(bank,warm['seed'],warm['asset_sha256'],warm['parameter_names'],warm['parameters'].numpy(),low,high,62)
    assert checked.shape==(60,32)
    sources={str(q.resolve()):hashlib.sha256(q.read_bytes()).hexdigest() for q in [a.checkpoint,a.bank,Path(__file__)]}
    warm['initialization_provenance']=sources;bank['preparation_family_provenance']=sources
    a.output.mkdir(parents=True);torch.save(warm,a.output/'policy.pt')
    (a.output/'CANDIDATE_BANK.json').write_text(json.dumps(bank,indent=2)+'\n')
    report=dict(created_at=datetime.now(timezone.utc).isoformat(),scope='CPU-only new preparation initialization; no native execution or inherited outcome.',
        configuration=config,reference_task_velocity_gain=.55,other_task_velocity_gain=.75,candidates=60,
        original_physics_actuator_limits_and_success_flags_unchanged=True,native_execution=False,formal_acceptance=False,
        policy_sha256=hashlib.sha256((a.output/'policy.pt').read_bytes()).hexdigest(),
        bank_sha256=hashlib.sha256((a.output/'CANDIDATE_BANK.json').read_bytes()).hexdigest(),sources=sources)
    (a.output/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
