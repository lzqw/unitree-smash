#!/usr/bin/env python3
"""Seed bounded pose/lead proposals around the inward string initialization."""
import argparse,hashlib,json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from quadsmash.control.full_racket_teacher_v4 import LOW,HIGH
from quadsmash.control.full_racket_recovery_search_v4 import RECOVERY_LOW,RECOVERY_HIGH
from quadsmash.control.full_racket_contact_normal_search_v4 import NORMAL_LOW,NORMAL_HIGH
from quadsmash.control.full_racket_candidate_bank_v4 import validate_candidate_bank


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--source-checkpoint',type=Path,required=True)
    p.add_argument('--fit-report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()
    target=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    original=torch.load(a.source_checkpoint,map_location='cpu',weights_only=False)
    report=json.loads(a.fit_report.read_text())
    assert not target['native_execution'] and original['native_execution']
    assert target['asset_sha256']==original['asset_sha256']
    assert target['seed']['string_uv_m']==[0.,.07]
    assert report['policy_sha256']==sha(a.checkpoint)
    assert target['source_policy_sha256']==sha(a.source_checkpoint)
    lower=np.array([*LOW,.8,-.12,*([-5.]*6),*RECOVERY_LOW,*NORMAL_LOW],dtype=np.float32)
    upper=np.array([*HIGH,1.45,.12,*([5.]*6),*RECOVERY_HIGH,*NORMAL_HIGH],dtype=np.float32)
    lower[3]=1.2;lower[22:24]=-.3;upper[22:24]=.3
    reference=target['parameters'].numpy().copy();assert reference.shape==(38,)
    delta=np.array(report['joint_offset_delta_rad'],dtype=np.float32)
    rows=[];labels=[]
    for fraction in [0.,.5,1.,1.5]:
        for lead_delta in [-.01,.01]:
            row=reference.copy()
            row[8:14]=original['parameters'][8:14].numpy()+fraction*delta
            row[14]=original['parameters'][14].item()+lead_delta
            rows.append(np.clip(row,lower,upper).tolist())
            labels.append(dict(retarget_fraction=fraction,arm_lead_delta_s=lead_delta))
    bank=dict(schema='full-racket-candidate-bank-v4',asset_sha256=target['asset_sha256'],
        seed=target['seed'],parameter_names=target['parameter_names'],
        reference_parameters=reference.tolist(),parameters=rows,labels=labels,
        scope='Unexecuted algorithm proposals only. Two exact warm references remain reserved; eight structured pose/lead proposals and six Gaussian proposals fit a16candidate search.',
        native_execution=False,formal_acceptance=False,source_sha256={str(p):sha(p) for p in
            [a.checkpoint,a.source_checkpoint,a.fit_report,Path(__file__)]})
    checked=validate_candidate_bank(bank,target['seed'],target['asset_sha256'],target['parameter_names'],
        reference,lower,upper,14)
    assert checked.shape==(8,38)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(bank,indent=2)+'\n')
    print(json.dumps(dict(output=str(a.output),rows=len(rows),sha256=sha(a.output),native_execution=False)))


if __name__=='__main__':main()
