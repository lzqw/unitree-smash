"""Score physical support and the frozen tangential model on new training-only native launches."""
import argparse
from collections import Counter
import math
import json
from pathlib import Path

import torch

from scripts.audit_supermimic_native_window_response import complete_window
from scripts.audit_supermimic_nominal_contact_repeat import window_force_balance
from quadsmash.rl.supermimic_action_identification_dataset_v1 import _load_branch
from scripts.calibrate_supermimic_toi_slip import TOISlip
from scripts.evaluate_supermimic_contact_rollout import sha
from scripts.fit_supermimic_native_racket_toi_head import free_rollout, g07
from scripts.declare_supermimic_contact_training_cases import resolve_probe_cases
from quadsmash.rl.supermimic_ball_impulse_prior_v1 import BallImpulsePriorV1


def supported_native(case):
    return (case['mode']=='racket_stable' and case['executed_target_change_rad']>0
            and math.dist(case['native_delta_mps'], (0.,0.,0.))>max(1e-8,3*case['repeat_noise_mps']))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('collection','fit','calibration','output'):
        parser.add_argument('--'+key,required=True,type=Path)
    args=parser.parse_args()
    torch.set_num_threads(2)
    read=lambda path: json.loads(path.read_text())
    startup=read(args.collection/'startup.json')
    declaration,=startup['specs']
    spec=declaration['spec']
    cases,case_hash=resolve_probe_cases(spec,[])
    if case_hash is None or sha(Path(declaration['path']))!=declaration['sha256']:
        raise ValueError('Training launches or probe specification changed')
    fit=read(args.fit)
    if sha(args.calibration)!=fit['calibration_sha256']:
        raise ValueError('Calibration hash mismatch')
    stat=read(args.collection/'status.json')
    for r in stat['completed']:
        folder=args.collection/r['spec']/r['branch']
        contract=read(folder/'task_contract.json')
        if (contract.get('contact_training_panel_sha256')!=case_hash or not contract.get('contact_training_only')
                or read(folder/'native_probe_manifest.json')['spec']!=spec):
            raise ValueError('Native recording was not made on the declared training launches')
    prior=BallImpulsePriorV1(normal_impulse_scale=read(args.calibration)['fitted']['normal_impulse_scale']).eval()
    _,_,noise,groups,seq,execution=g07(args.collection,prior)
    model=TOISlip(fit['selected']['friction'],fit['selected']['sticking'])
    score=free_rollout(model,prior,seq,groups,noise,execution)
    for case in score['cases']:
        case['case_id']=cases[case['lane']]['case_id']
    supported=[r['case_id'] for r in score['cases'] if supported_native(r)]
    physics_path=Path(__file__).resolve().parents[1]/'configs/shuttle_physics_v2.json'
    if sha(physics_path)!=fit['physics_sha256']:
        raise ValueError('Shuttle mass, drag or solver timebase changed')
    physics=read(physics_path)
    balances={}
    for item in stat['completed']:
        branch=item['branch']
        run=_load_branch(args.collection/item['spec'],branch)
        windows=complete_window(run['raw'],spec,prior.steps)
        balances[branch]=window_force_balance(windows,physics)
    result=dict(training_only=True,policy_training_admitted=False,contact_receipt=False,
                case_sha256=case_hash,source_sha256=startup['source_sha256'],status_sha256=sha(args.collection/'status.json'),
                physics_fit_sha256=sha(args.fit),counts=dict(Counter(groups)),independent_stable_with_response=supported,
                native_racket=score,filtered_momentum_balance=balances,unchanged_execution_lanes=execution['unchanged_lanes'],
                actor_updates=0,v13_updates=0)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)


if __name__=='__main__':
    main()
