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
from quadsmash.rl.supermimic_toi_racket_objective import bind_intercept_reference, preview_intercept_lanes
from scripts.evaluate_supermimic_fresh_wm_panel import digest
from scripts.train_supermimic_physical_block import select_launch_for_start


@torch.no_grad()
def main():
    cli=argparse.ArgumentParser(description=__doc__)
    for k in ('checkpoint','reference-bank','shadow','output'):
        cli.add_argument('--'+k,type=Path,required=True)
    cli.add_argument('--online-guard',type=Path,help='Use the checked live WM and preview the actual gate at the intercept')
    cli.add_argument('--compare-ticks',nargs='*',type=int,default=[])
    args=cli.parse_args()
    if args.compare_ticks and not args.online_guard:
        cli.error('--compare-ticks requires --online-guard')
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
    actor,ad,template,launch,manifest=load_physical_loop(archive=option('archive'),safety_mapping=option('safety-mapping'),
        physical_checkpoint=paths['physical'],ball_checkpoint=paths['ball'],task_checkpoint=paths['task'],
        action_ball_checkpoint=paths.get('action_ball'),execution_root=meta['g07_execution_root'],device='cpu')
    original=TaskSpaceReferenceBankV1.from_payload(torch.load(args.reference_bank,weights_only=True,map_location='cpu'))
    if ad.manifest['checkpoint_sha256']!=meta['checkpoint_sha256']:
        raise ValueError('Changed world dependencies')
    saved=ck['actor_start_bank']
    bank=ActorStartBank(capacity=saved['capacity'])
    bank.load_state_dict(saved)
    offset=(ck['config'].get('intercept_plane_calibration') or {}).get('offset_m',0.)
    held=ck['config']['wm_holdout_lanes']
    original.assert_excludes(held)
    trainer=None
    world_identity=None
    compared={}
    if args.online_guard:
        from quadsmash.rl.supermimic_physical_learner_v1 import PhysicalHorizonTrainer
        from quadsmash.rl.supermimic_timebase_v2 import Timebase
        guard=json.loads(args.online_guard.read_text().splitlines()[-1])
        world_identity=tensor_digest(ck['world'].items())
        if not guard['passed'] or guard['world_sha256']!=world_identity or guard['world_updates']!=ck['world_updates']:
            raise ValueError('Preview requires the checked live world')
        actor.load_state_dict(ck['actor'])
        ad.load_state_dict(ck['world'])
        ad.set_contact_bptt_mode('detached')
        row=bank.entries[0]['start']
        initial=PhysicalImaginationStateV1.from_record(row,executor_template=template,launch=select_launch_for_start(row,launch),device='cpu')
        timebase=Timebase(**json.loads((Path(__file__).resolve().parents[1]/'configs/supermimic_task_horizon_v2.json').read_text()))
        if timebase.imagination_steps!=shadow['imagination_steps']:
            raise ValueError('Mismatched native shadow horizon')
        trainer=PhysicalHorizonTrainer(actor,ad,initial,timebase,block_name='approach',reference_bank=original,approach_objective='toi_racket_only',toi_plane_offset_m=offset)
        trainer.critic.load_state_dict(ck['critic'])
    masks={}
    cases=[]
    for entry in bank.entries:
        row=entry['start']
        state=PhysicalImaginationStateV1.from_record(row,executor_template=template,launch=select_launch_for_start(row,launch),device='cpu')
        ok=training_start_mask(row,held)
        ref=bind_intercept_reference(state,original,ad.physical_model.geometry,shadow['imagination_steps']*manifest['task_contract']['control_dt_s'],plane_offset_m=offset)
        valid=ok & ref['matched'].cpu().reshape(-1) if ref is not None else torch.zeros_like(ok)
        if trainer is not None:
            state.clock_goal_context['actor_lane_mask']=ok
            if ref is not None:
                valid=preview_intercept_lanes(state,ref,valid,trainer)
            if entry['native_tick'] in args.compare_ticks:
                objective=trainer._rollout_objective(state)
                weights=objective['toi_weights']
                full=(weights>0).any(0) if weights is not None else torch.zeros_like(ok)
                if not torch.equal(full,valid):
                    raise AssertionError('Start eligibility differs from the production TOI objective')
                compared[entry['native_tick']]={k:int((full&m).sum()) for k,m in entry['masks'].items()}
        masks[id(row)]=valid
        cases.append(dict(native_tick=entry['native_tick'],policy_version=entry['policy_version'],stream=entry['stream'],
            categories={name:dict(archived=int(m.sum()),training=int((m&ok).sum()),valid=int((m&valid).sum()))
                        for name,m in entry['masks'].items()}))
    if set(compared)!=set(args.compare_ticks):
        raise ValueError('Requested tick is absent from the saved bank')
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
    if trainer is not None:
        result.update(world_sha256=world_identity,full_objective_comparisons=compared,
                      selection_criterion='positive lane weight at the predicted arrival in the live WM')
    with args.output.open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False)


if __name__=='__main__':
    main()
