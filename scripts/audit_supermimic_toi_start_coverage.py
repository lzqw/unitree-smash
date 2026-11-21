"""Audit the actual saved Approach reservoir against the causal TOI training objective."""
import argparse
import json
from pathlib import Path

import torch

from quadsmash.rl.supermimic_actor_start_bank_v1 import ActorStartBank, CATEGORIES
from quadsmash.rl.supermimic_physical_imagination_state_v1 import PhysicalImaginationStateV1
from quadsmash.rl.supermimic_physical_loop_loading_v1 import load_physical_loop, tensor_digest
from quadsmash.rl.supermimic_physical_start_cache_v1 import training_start_mask
from quadsmash.rl.supermimic_task_space_reference_v1 import TaskSpaceReferenceBankV1
from quadsmash.rl.supermimic_toi_racket_objective import bind_intercept_reference
from scripts.evaluate_supermimic_fresh_wm_panel import digest
from scripts.train_supermimic_physical_block import select_launch_for_start


@torch.no_grad()
def main():
    cli=argparse.ArgumentParser(description=__doc__)
    for k in ('checkpoint','reference-bank','shadow','output'):
        cli.add_argument('--'+k,type=Path,required=True)
    args=cli.parse_args()
    torch.set_num_threads(2)
    ck=torch.load(args.checkpoint,mmap=True,map_location='cpu',weights_only=False)
    shadow=json.loads(args.shadow.read_text())
    if (not shadow['passed'] or tensor_digest(ck['actor'].items())!=shadow['actor_sha256']
            or digest(args.reference_bank)!=shadow['reference_bank_sha256']):
        raise ValueError('Different native shadow, actor or target reference bank')
    argv=json.loads((args.checkpoint.parent/'startup.json').read_text())['argv']
    def option(key):
        return Path(argv[argv.index('--'+key)+1])
    meta=ck['metadata']['physical_imagination']
    paths=meta['checkpoint_paths']
    _,ad,template,launch,manifest=load_physical_loop(archive=option('archive'),safety_mapping=option('safety-mapping'),
        physical_checkpoint=paths['physical'],ball_checkpoint=paths['ball'],task_checkpoint=paths['task'],
        action_ball_checkpoint=paths.get('action_ball'),execution_root=meta['g07_execution_root'],device='cpu')
    original=TaskSpaceReferenceBankV1.from_payload(torch.load(args.reference_bank,weights_only=True,map_location='cpu'))
    if ad.manifest['checkpoint_sha256']!=meta['checkpoint_sha256']:
        raise ValueError('Changed world dependencies')
    saved=ck['actor_start_bank']
    bank=ActorStartBank(capacity=saved['capacity'])
    bank.load_state_dict(saved)
    held=ck['config']['wm_holdout_lanes']
    original.assert_excludes(held)
    masks={}
    cases=[]
    for entry in bank.entries:
        row=entry['start']
        state=PhysicalImaginationStateV1.from_record(row,executor_template=template,launch=select_launch_for_start(row,launch),device='cpu')
        ok=training_start_mask(row,held)
        ref=bind_intercept_reference(state,original,ad.physical_model.geometry,shadow['imagination_steps']*manifest['task_contract']['control_dt_s'])
        valid=ok & ref['matched'].cpu().reshape(-1) if ref is not None else torch.zeros_like(ok)
        masks[id(row)]=valid
        cases.append(dict(native_tick=entry['native_tick'],policy_version=entry['policy_version'],stream=entry['stream'],
            categories={name:dict(archived=int(m.sum()),training=int((m&ok).sum()),valid=int((m&valid).sum()))
                        for name,m in entry['masks'].items()}))
    selections=[]
    for attempt in range(1,21):
        row,report=bank.sample(attempt,eligible_lanes=lambda r:masks[id(r)])
        if row is not None and not training_start_mask(row,held).equal(row['_actor_lane_mask']):
            raise AssertionError('Selected a held lane')
        selections.append(report)
    result=dict(actor_sha256=shadow['actor_sha256'],reference_bank_sha256=shadow['reference_bank_sha256'],
                checkpoint_sha256=digest(args.checkpoint),entries=cases,selections=selections,
                totals={name:{k:sum(r['categories'][name][k] for r in cases) for k in ('archived','training','valid')}
                        for name in CATEGORIES},actor_updates=0)
    with args.output.open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False)


if __name__=='__main__':
    main()
