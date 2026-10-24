#!/usr/bin/env python3
"""Prepare a bounded lead/cartesian-gain/normal bank around a native attack."""
import argparse, copy, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from quadsmash.control.full_racket_candidate_bank_v4 import validate_candidate_bank
from quadsmash.control.full_racket_contact_normal_search_v4 import NORMAL_LOW,NORMAL_HIGH
from quadsmash.control.full_racket_parameter_layout_v4 import controller_parameter_names
from quadsmash.control.full_racket_recovery_search_v4 import RECOVERY_LOW,RECOVERY_HIGH
from quadsmash.control.full_racket_teacher_v4 import LOW,HIGH

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 args=p.parse_args()
 if args.output.exists():raise ValueError('fresh output required')
 source=torch.load(args.checkpoint,map_location='cpu',weights_only=True);assert len(source['parameters'])==38
 names=controller_parameter_names(search_arm_velocity=True,search_recovery_parameters=True,search_contact_normal=True);assert source['parameter_names']==names
 ref=source['parameters'].numpy().copy();lower=np.array(LOW+[.8,-.12]+[-5.]*6+RECOVERY_LOW+NORMAL_LOW,dtype=np.float32);upper=np.array(HIGH+[1.45,.12]+[5.]*6+RECOVERY_HIGH+NORMAL_HIGH,dtype=np.float32);lower[3]=1.2;lower[22:24],upper[22:24]=-.3,.3
 leads=[.02,.04,.06,.08];gains=[.28,.34,.40,.46,.52];normals=[.08,.10,.12];rows=[];labels=[]
 for lead in leads:
  for gain in gains:
   for normal in normals:
    x=ref.copy();x[14]=lead;x[18]=gain;x[37]=normal;rows.append(x);labels.append(dict(arm_lead=lead,cartesian_gain=gain,normal_bias=normal,performance_inherited=False))
 rows=np.asarray(rows,dtype=np.float32);assert rows.shape==(60,38);fixed=[i for i in range(38) if i not in [14,18,37]];np.testing.assert_array_equal(rows[:,fixed],np.tile(ref[fixed],(60,1)))
 warm=copy.deepcopy(source);warm.pop('training_result',None);warm.update(native_execution=False,performance_inherited=False,formal_acceptance=False,kind='Unexecuted contact lead/cartesian gain/normal bank; fresh native measurement required.')
 bank=dict(schema='full-racket-candidate-bank-v4',asset_sha256=warm['asset_sha256'],seed=warm['seed'],parameter_names=names,reference_parameters=ref.tolist(),parameters=rows.tolist(),proposal_labels=labels,controller_revision='contact-lead-gain-normal-v1',native_execution=False,performance_inherited=False);validate_candidate_bank(bank,warm['seed'],warm['asset_sha256'],names,ref,lower,upper,62)
 args.output.mkdir(parents=True);torch.save(warm,args.output/'policy.pt');(args.output/'CANDIDATE_BANK.json').write_text(json.dumps(bank,indent=2)+'\n');paths=[args.checkpoint,Path(__file__)];report=dict(created_at=datetime.now(timezone.utc).isoformat(),scope='CPU-prepared bank only; changes existing arm lead/cartesian gain/contact normal coordinates, keeps intercept plane1.05 and all gates.',candidates=60,changed_coordinates=[14,18,37],leads=leads,cartesian_gains=gains,normals=normals,native_execution=False,performance_inherited=False,formal_acceptance=False,source_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},policy_sha256=hashlib.sha256((args.output/'policy.pt').read_bytes()).hexdigest(),bank_sha256=hashlib.sha256((args.output/'CANDIDATE_BANK.json').read_bytes()).hexdigest());(args.output/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
