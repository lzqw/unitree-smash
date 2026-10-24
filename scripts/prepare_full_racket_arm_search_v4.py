#!/usr/bin/env python3
"""Prepare traceable unexecuted32-coordinate search initializations."""
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
from quadsmash.control.full_racket_arm_search_v4 import extend_arm_velocity_parameters,VELOCITY_PARAMETER_NAMES
from quadsmash.control.full_racket_teacher_v4 import PARAMETER_NAMES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--legacy-source',type=Path,required=True)
    p.add_argument('--body-bank',type=Path,required=True)
    p.add_argument('--pitch-tradeoff',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Use a fresh initialization directory')
    source=torch.load(a.source,map_location='cpu',weights_only=True)
    legacy=torch.load(a.legacy_source,map_location='cpu',weights_only=True)
    body=json.loads(a.body_bank.read_text())
    tradeoff=json.loads(a.pitch_tradeoff.read_text())
    assert source['asset_sha256']==legacy['asset_sha256']==body['asset_sha256']
    assert source['seed']['joint_position_rad']==legacy['seed']['joint_position_rad']
    velocities=[legacy['seed']['joint_velocity_seed_rad_s'],source['seed']['joint_velocity_seed_rad_s']]
    for weight in [.3,1.]:
        velocity=list(next(r for r in tradeoff['rows'] if r['body_weight']==weight)['qdot'])
        velocity[-1]=0.
        velocities.append(velocity)
    assert np.asarray(velocities).shape==(4,6) and np.max(np.abs(velocities))<=5.
    paths=[a.source,a.legacy_source,a.body_bank,a.pitch_tradeoff,Path(__file__)]
    provenance={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    a.output.mkdir(parents=True)
    variants=[]
    for mode,gain,run_name in [('velocity_only',0.,'ball_velocity32_cem05'),
                                ('causal_prepose',.5,'ball_prepose_velocity32_cem06')]:
        destination=a.output/mode
        destination.mkdir()
        checkpoint=copy.deepcopy(source)
        checkpoint['parameters']=extend_arm_velocity_parameters(source['parameters'],source['seed']['joint_velocity_seed_rad_s'])
        if gain:checkpoint['seed']=dict(checkpoint['seed'],causal_arm_prepose_gain=gain)
        checkpoint.update(schema='full-racket-arm-search-initialization-v4',
                          parameter_names=PARAMETER_NAMES+['intercept_plane_x','strike_timing_residual']+VELOCITY_PARAMETER_NAMES,
                          kind='CPU prepared initialization; no new native execution or inherited fitness',
                          formal_acceptance=False,native_execution=False,initialization_provenance=provenance)
        checkpoint.pop('training_result',None)
        torch.save(checkpoint,destination/'policy.pt')
        velocity_bank=dict(asset_sha256=checkpoint['asset_sha256'],reference_seed=checkpoint['seed'],
                           joint_velocity_seeds_rad_s=velocities,source_provenance=provenance,
                           scope='Four unexecuted directional proposals, spanning old and instantaneous momentum/pitch objectives; each native outcome must be remeasured.',
                           native_execution=False,formal_acceptance=False)
        (destination/'VELOCITY_BANK.json').write_text(json.dumps(velocity_bank,indent=2)+'\n')
        recovery=copy.deepcopy(body)
        recovery['seed']=checkpoint['seed']
        recovery['scope']='Historical native body parameters are initialization proposals under changed arm control; no recovery or contact outcome transfers.'
        recovery['native_execution_with_new_arm_seed']=False
        recovery['performance_inherited']=False
        (destination/'RECOVERY_BANK.json').write_text(json.dumps(recovery,indent=2)+'\n')
        relative=str(destination.resolve().relative_to(ROOT))
        command=[sys.executable,'scripts/run_gpu_job.py','--goal','G07','--budget-window','g07-full-racket-unlimited-20261024',
                 '--max-seconds','5400','--env-steps-cap','204800','--output','reports/G07_full_racket_v4/'+run_name,
                 '--require-artifact','completed.json','--',sys.executable,'scripts/search_full_racket_ball_v4.py',
                 '--output','reports/G07_full_racket_v4/'+run_name,'--warm-start',relative+'/policy.pt',
                 '--candidates','64','--generations','2','--sigma-fraction','.05','--training-indices','1','65',
                 '--continue-after-miss','--contact-curriculum','--search-arm-velocity',
                 '--arm-velocity-bank',relative+'/VELOCITY_BANK.json','--recovery-leg-bank',relative+'/RECOVERY_BANK.json',
                 '--push-search-range','.30','--headless']
        variants.append(dict(mode=mode,prepose_gain=gain,checkpoint=relative+'/policy.pt',
                             checkpoint_sha256=hashlib.sha256((destination/'policy.pt').read_bytes()).hexdigest(),
                             prepared_command=command,command_status='NOT_RUN; use only after merging code into the main workspace and checking the existing GPU guard.'))
    report=dict(created_at=datetime.now(timezone.utc).isoformat(),scope='CPU-only future controller initializations; no native policy result.',
                variants=variants,sources=provenance,formal_acceptance=False,native_execution=False)
    (a.output/'INITIALIZATIONS.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(variants=[dict(mode=x['mode'],checkpoint_sha256=x['checkpoint_sha256']) for x in variants],native_execution=False)))


if __name__=='__main__':main()
