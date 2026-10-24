#!/usr/bin/env python3
"""Prepare bounded normal/timing/swing-gain proposals around an actual37policy."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from quadsmash.control.full_racket_contact_normal_search_v4 import (
    extend_contact_normal_parameters,NORMAL_LOW,NORMAL_HIGH)
from quadsmash.control.full_racket_parameter_layout_v4 import controller_parameter_names
from quadsmash.control.full_racket_candidate_bank_v4 import validate_candidate_bank
from quadsmash.control.full_racket_teacher_v4 import LOW,HIGH
from quadsmash.control.full_racket_recovery_search_v4 import RECOVERY_LOW,RECOVERY_HIGH
from quadsmash.control.full_racket_contact_curriculum_v4 import validate_recovery_search_min


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--recovery-search-min',type=float,default=1.20)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Use a fresh output directory')
    minimum=validate_recovery_search_min(a.recovery_search_min)
    source=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    assert len(source['parameters'])==37
    assert source['parameter_names']==controller_parameter_names(search_arm_velocity=True,search_recovery_parameters=True)
    reference=extend_contact_normal_parameters(source['parameters']).numpy()
    names=controller_parameter_names(search_arm_velocity=True,search_recovery_parameters=True,search_contact_normal=True)
    lower=np.array(LOW+[.8,-.12]+[-5.]*6+RECOVERY_LOW+NORMAL_LOW,dtype=np.float32)
    upper=np.array(HIGH+[1.45,.12]+[5.]*6+RECOVERY_HIGH+NORMAL_HIGH,dtype=np.float32)
    lower[22:24],upper[22:24],lower[3]=-.3,.3,minimum
    rows,labels=[],[]
    for angle in [-.04,0.,.02,.04,.06,.08]:
        for lead_delta in [-.02,-.01,0.,.01,.02]:
            for multiplier in [.9,1.]:
                row=reference.copy()
                row[37]=angle
                row[14]=np.clip(reference[14]+lead_delta,lower[14],upper[14])
                row[7]=np.clip(reference[7]*multiplier,lower[7],upper[7])
                rows.append(row)
                labels.append(dict(contact_normal_downward_pitch_bias_rad=angle,
                    arm_lead_delta_s=lead_delta,arm_velocity_gain_multiplier=multiplier,
                    performance_inherited=False))
    rows=np.stack(rows);assert rows.shape==(60,38)
    assert len(set(tuple(x) for x in rows))==60,'Reference too near bounds for this local grid'
    fixed=[i for i in range(37) if i not in [7,14]]
    np.testing.assert_array_equal(rows[:,fixed],np.tile(reference[fixed],(60,1)))
    assert (abs(rows[:,26:32]*rows[:,7:8])<=5.).all()
    warm=copy.deepcopy(source);warm['parameters']=torch.tensor(reference);warm['parameter_names']=names
    warm.pop('training_result',None)
    warm.update(native_execution=False,performance_inherited=False,formal_acceptance=False,
                kind='Unexecuted38-coordinate face/timing/gain proposals. Target direction is a bounded robot command, not assigned ball velocity; all physical outcomes must be remeasured.')
    bank=dict(schema='full-racket-candidate-bank-v4',asset_sha256=warm['asset_sha256'],seed=warm['seed'],
              parameter_names=names,reference_parameters=reference.tolist(),parameters=rows.tolist(),
              proposal_labels=labels,controller_revision='bounded-contact-normal38-v1',
              native_execution=False,performance_inherited=False)
    validate_candidate_bank(bank,warm['seed'],warm['asset_sha256'],names,reference,lower,upper,62)
    paths=[a.checkpoint,Path(__file__)]+[ROOT/'quadsmash/control'/n for n in [
        'full_racket_teacher_v4.py','full_racket_parameter_layout_v4.py',
        'full_racket_contact_normal_search_v4.py','full_racket_recovery_search_v4.py','full_racket_arm_search_v4.py']]
    provenance={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    warm['contact_normal_search_provenance']=provenance;bank['contact_normal_search_provenance']=provenance
    a.output.mkdir(parents=True)
    torch.save(warm,a.output/'policy.pt')
    (a.output/'CANDIDATE_BANK.json').write_text(json.dumps(bank,indent=2)+'\n')
    report=dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='Unexecuted optional38coordinate normal/timing/gain grid. Same standing37reference coordinates; no new ball path, physical joint, asset or acceptance gate.',
        parameter_count=38,candidates=60,first37_changed_coordinates=[7,14],
        appended_coordinate_name=names[-1],normal_bias_bounds_rad=[NORMAL_LOW[0],NORMAL_HIGH[0]],
        desired_normal_downward_bias_grid_rad=[-.04,0.,.02,.04,.06,.08],
        source_native_execution=source.get('native_execution',False),
        source_actual_training_rows=[{k:r[k] for k in ['source_training_index','attack_outgoing','airborne_string_hit','recovery_success','landed_target','complete_smash']} for r in source.get('training_result',[])],
        new_policy_performance_inherited=False,all_existing_recovery_coordinates_preserved=True,
        all_physical_assets_hardware_limits_and_original_task_gates_unchanged=True,
        recovery_search_min_local_s=minimum,recommended_search_coordinates=[7,14,37],
        required_flags=['--search-arm-velocity','--search-recovery-parameters','--search-contact-normal'],
        native_execution=False,formal_acceptance=False,source_sha256=provenance,
        policy_sha256=hashlib.sha256((a.output/'policy.pt').read_bytes()).hexdigest(),
        bank_sha256=hashlib.sha256((a.output/'CANDIDATE_BANK.json').read_bytes()).hexdigest())
    (a.output/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'}),flush=True)


if __name__=='__main__':main()
