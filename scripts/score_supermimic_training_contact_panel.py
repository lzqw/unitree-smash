"""Score physical support and the frozen tangential model on new training-only native launches."""
import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from scripts.calibrate_supermimic_toi_slip import TOISlip
from scripts.evaluate_supermimic_contact_rollout import sha
from scripts.fit_supermimic_native_racket_toi_head import free_rollout, g07
from scripts.declare_supermimic_contact_training_cases import resolve_probe_cases
from quadsmash.rl.supermimic_ball_impulse_prior_v1 import BallImpulsePriorV1


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
    supported=[r['case_id'] for r in score['cases'] if r['mode']=='racket_stable'
               and r['executed_target_change_rad']>0 and r['candidate']['cosine'] is not None]
    result=dict(training_only=True,policy_training_admitted=False,contact_receipt=False,
                case_sha256=case_hash,source_sha256=startup['source_sha256'],status_sha256=sha(args.collection/'status.json'),
                physics_fit_sha256=sha(args.fit),counts=dict(Counter(groups)),independent_stable_with_response=supported,
                native_racket=score,actor_updates=0,v13_updates=0)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)


if __name__=='__main__':
    main()
