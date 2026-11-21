"""Score two Approach policies on disjoint held cases using the proposal WM and reference bank."""
import argparse
import json
from pathlib import Path

import torch

from quadsmash.rl.supermimic_native_pair_capture_v1 import tree_digest
from quadsmash.rl.supermimic_physical_imagination_state_v1 import PhysicalImaginationStateV1
from quadsmash.rl.supermimic_physical_learner_v1 import PhysicalHorizonTrainer
from quadsmash.rl.supermimic_physical_loop_loading_v1 import load_physical_loop, tensor_digest
from quadsmash.rl.supermimic_physical_start_cache_v1 import APPROACH_TICKS, load_start_cache
from quadsmash.rl.supermimic_task_space_reference_v1 import TaskSpaceReferenceBankV1
from quadsmash.rl.supermimic_timebase_v2 import Timebase
from quadsmash.rl.supermimic_toi_racket_objective import bind_intercept_reference
from scripts.compare_supermimic_fixed_stages import load_run
from scripts.evaluate_supermimic_fresh_wm_panel import digest


def arrival_state(trainer, initial, lane, step):
    state=initial.branch()
    cfg=trainer.timebase
    ref=bind_intercept_reference(state,trainer.reference_bank,trainer.adapter.physical_model.geometry,
                                 cfg.control_dt_s*cfg.imagination_steps)
    for _ in range(step):
        action=trainer._actor_action(state,sample=False)[0]
        state,_,_,_=trainer.adapter(state,action,detach_ball=True)
    return dict(step=step,elapsed=float(state.clock_goal_context['elapsed'].reshape(-1)[lane]),
                arrival=float(ref['intercept']['time'][lane]),gate=float(trainer._head_gate(state)[lane]),
                had_contact=bool(state.task_memory['rally']['had_contact'].reshape(-1)[lane]>.5))


@torch.no_grad()
def audit(args):
    read = lambda path: json.loads(path.read_text())
    def checkpoint(path):
        return torch.load(path, mmap=True, weights_only=False, map_location='cpu')
    before, after = checkpoint(args.before), checkpoint(args.after)
    rows, identity, initial_actor = load_run(args.baseline)
    rows = sorted(rows, key=lambda row: row['env_id'])
    actor_hash = tensor_digest(after['actor'].items())
    world_hash = tensor_digest(after['world'].items())
    checks = [json.loads(line) for line in args.guard.read_text().splitlines()]
    if not any(r['passed'] and r['world_sha256']==world_hash and r['world_updates']==after['world_updates'] for r in checks):
        raise ValueError('The proposal WM lacks a matching forward check')
    changed = [k for k,v in before['actor'].items() if not torch.equal(v, after['actor'][k])]
    if (not changed or any(not k.startswith('actor.approach.') for k in changed)
            or initial_actor!=tensor_digest(before['actor'].items())
            or tree_digest(before['reference_bank'])!=tree_digest(after['reference_bank'])):
        raise ValueError('Expected the matched initial policy, fixed targets and an Approach-only candidate')
    meta = after['metadata']['physical_imagination']
    paths = meta['checkpoint_paths']
    actor, adapter, template, launch, manifest = load_physical_loop(archive=args.archive, safety_mapping=args.safety_mapping,
        physical_checkpoint=paths['physical'], ball_checkpoint=paths['ball'],task_checkpoint=paths['task'],
        action_ball_checkpoint=paths.get('action_ball'), execution_root=meta['g07_execution_root'],device='cpu')
    initial_hash = digest(args.archive.parent/'initial_states.json')
    if (initial_hash!=digest(args.baseline/'initial_states.json') or identity['seed']!=manifest['seed']
            or manifest['policy_execution']!=identity['execution'] or identity['task']!=manifest['task_contract']
            or adapter.manifest['checkpoint_sha256']!=meta['checkpoint_sha256']):
        raise ValueError('Archive start cases or execution contract differ')
    adapter.load_state_dict(after['world'])
    adapter.set_contact_bptt_mode('detached')
    timebase = Timebase(**read(Path(__file__).resolve().parents[1]/'configs/supermimic_task_horizon_v2.json'))
    held = after['config']['wm_holdout_lanes']
    ref = TaskSpaceReferenceBankV1.from_payload(after['reference_bank'])
    ref.assert_excludes(held)
    starts = load_start_cache(args.archive,manifest)
    result = {}
    trainer = None
    for qualified in (False,True):
        eligible = torch.tensor([r['env_id'] in held and bool(r['qualified_airborne_string_contact'])==qualified for r in rows])
        initial_count = int(eligible.sum())
        samples = []
        for tick in APPROACH_TICKS:
            state = PhysicalImaginationStateV1.from_record(starts[tick],executor_template=template,launch=launch,device='cpu')
            selected = eligible & ~state.task_memory['rally']['had_contact'].reshape(-1).bool()
            episodes = state.clock_goal_context['episode_index'].reshape(-1)
            if any(rows[i]['episode_index']!=int(episodes[i]) for i in torch.where(selected)[0].tolist()):
                raise ValueError('Fixed case starts belong to different episodes')
            if not selected.any():
                continue
            state.clock_goal_context['actor_lane_mask'] = selected
            if trainer is None:
                trainer = PhysicalHorizonTrainer(actor,adapter,state,timebase,block_name='approach',
                    reference_bank=ref,approach_objective='toi_racket_only',toi_racket_weight=after['config']['toi_racket_weight'])
                trainer.critic.load_state_dict(after['critic'])
            actor.load_state_dict(before['actor'])
            old = trainer._rollout_objective(state)
            w = old['toi_weights']
            if w is None or not w.any():
                continue
            used = w.gt(0).any(0)
            eligible &= ~used
            actor.load_state_dict(after['actor'])
            new = trainer._rollout_objective(state,effective_weights=old['effective_weights'],baseline_toi_weights=w)
            retained = new['toi_weights'].gt(0).any(0)
            ids = torch.where(used)[0].tolist()
            lost=torch.where(used & ~retained)[0].tolist()
            common=used & retained
            common_weights=w*common
            on_common={}
            if lost:
                for name,ck in [('before',before),('after',after)]:
                    actor.load_state_dict(ck['actor'])
                    objective=trainer._rollout_objective(state,effective_weights=old['effective_weights'],
                                                          baseline_toi_weights=common_weights)
                    if not objective['toi_coverage_preserved']:
                        raise RuntimeError('Shared arrival coverage changed during deterministic replay')
                    on_common[name]=float(objective['tracking'])
                    on_common[name+'_lost_arrivals']={lane:arrival_state(trainer,state,lane,int(w[:,lane].argmax())) for lane in lost}
            else:
                on_common=dict(before=float(old['tracking']),after=float(new['tracking']))
            samples.append(dict(tick=tick,lanes=ids,case_ids=[rows[i]['case_id'] for i in ids],
                                old_weight=float(w.sum()),new_weight=float(new['toi_weights'].sum()),
                                lost_lanes=lost,common_weight=float(common_weights.sum()),common=on_common,
                                before=float(old['tracking']),after=float(new['tracking']),
                                coverage_preserved=new['toi_coverage_preserved']))
        matched = initial_count-int(eligible.sum())
        if matched!=sum(len(s['case_ids']) for s in samples):
            raise RuntimeError('Duplicate held case in imagined TOI measurement')
        summary = dict(eligible=initial_count,matched=matched,unmatched_lanes=torch.where(eligible)[0].tolist(),starts=samples)
        if matched and all(s['coverage_preserved'] for s in samples):
            for key in ('before','after'):
                summary[key]=sum(s['old_weight']*s[key] for s in samples)/sum(s['old_weight'] for s in samples)
        result['qualified' if qualified else 'missed'] = summary
    return dict(source_actor=initial_actor,candidate_actor=actor_hash,world_sha256=world_hash,world_updates=after['world_updates'],
                reference_sha256=tree_digest(after['reference_bank']),initial_states_sha256=initial_hash,
                archive_manifest_sha256=digest(args.archive/'manifest.json'),native_baseline_sha256=digest(args.baseline/'episodes_summary.json'),
                held_lanes=held,results=result,optimizer_updates=0,
                scope='Development prediction on identical recorded precontact starts. Each lane contributes at its earliest eligible baseline arrival and retains baseline weights. Native baseline outcome is used only for reporting strata.')


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ('before','after','archive','baseline','safety-mapping','guard','output'):
        cli.add_argument('--'+name,type=Path,required=True)
    args=cli.parse_args()
    torch.set_num_threads(3)
    result=audit(args)
    with args.output.open('x') as out:
        json.dump(result,out,indent=2,allow_nan=False)
        out.write('\n')
