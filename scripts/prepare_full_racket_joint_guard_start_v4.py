#!/usr/bin/env python3
"""Prepare unexecuted bounded joint preparation, preserving measured arm seeds."""
import argparse
import copy
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from quadsmash.control.full_racket_joint_preparation_v4 import validate_joint_preparation
from quadsmash.control.full_racket_candidate_bank_v4 import validate_candidate_bank
from quadsmash.control.full_racket_teacher_v4 import LOW,HIGH


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--anticipate-measured-motion',action='store_true')
    p.add_argument('--protect-post-hit-net',action='store_true')
    p.add_argument('--maximum-predictive-margin',type=float,default=.15)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Use a fresh output directory')
    source=torch.load(a.checkpoint,map_location='cpu',weights_only=True);bank=json.loads(a.bank.read_text())
    assert bank['seed']==source['seed'] and len(source['parameters'])==32
    config=validate_joint_preparation(dict(anticipate_measured_motion=a.anticipate_measured_motion,
        protect_post_hit_net_volume=a.protect_post_hit_net,
        maximum_predictive_margin_m=a.maximum_predictive_margin))
    warm=copy.deepcopy(source);warm['seed']=dict(source['seed'],joint_preparation_guard=config)
    warm['seed'].pop('cartesian_preparation',None)
    warm.pop('training_result',None)
    warm.update(kind='Unexecuted bounded joint preparation; no native contact, power or recovery outcome inherited.',
                native_execution=False,performance_inherited=False,formal_acceptance=False)
    revision='measured-tool-motion-and-posthit-net-v2' if a.anticipate_measured_motion or a.protect_post_hit_net else 'physical-support-joint-preparation-v1'
    bank=copy.deepcopy(bank);bank.update(seed=warm['seed'],controller_revision=revision,
        native_execution=False,scope='Original body and arm proposals with bounded own-side joint-command preparation; no outcomes inherited.')
    low=np.array(LOW+[.8,-.12]+[-5.]*6,dtype=np.float32);high=np.array(HIGH+[1.45,.12]+[5.]*6,dtype=np.float32)
    low[22:24]=-.3;high[22:24]=.3
    checked=validate_candidate_bank(bank,warm['seed'],warm['asset_sha256'],warm['parameter_names'],warm['parameters'].numpy(),low,high,62)
    assert checked.shape==(60,32) and torch.equal(warm['parameters'],source['parameters'])
    sources={str(q.resolve()):hashlib.sha256(q.read_bytes()).hexdigest() for q in [a.checkpoint,a.bank,Path(__file__)]}
    warm['initialization_provenance']=sources;bank['joint_preparation_provenance']=sources
    a.output.mkdir(parents=True);torch.save(warm,a.output/'policy.pt')
    (a.output/'CANDIDATE_BANK.json').write_text(json.dumps(bank,indent=2)+'\n')
    report=dict(created_at=datetime.now(timezone.utc).isoformat(),scope='CPU-only new preparation family; native outcomes must be remeasured.',
        configuration=config,original_all32reference_parameters_preserved=True,original_all60proposal_parameters_preserved=True,
        physical_support_includes_clamp=True,native_execution=False,formal_acceptance=False,
        policy_sha256=hashlib.sha256((a.output/'policy.pt').read_bytes()).hexdigest(),
        bank_sha256=hashlib.sha256((a.output/'CANDIDATE_BANK.json').read_bytes()).hexdigest(),sources=sources)
    (a.output/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
