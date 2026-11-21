"""Compare H64 predicted Recovery-policy changes with existing fixed-case native discounted returns."""
import argparse
import json
from pathlib import Path

import torch

from quadsmash.rl.supermimic_physical_imagination_state_v1 import PhysicalImaginationStateV1
from quadsmash.rl.supermimic_physical_learner_v1 import PhysicalHorizonTrainer
from quadsmash.rl.supermimic_physical_loop_loading_v1 import RECOVERY_NAMES, load_physical_loop, tensor_digest
from quadsmash.rl.supermimic_physical_start_cache_v1 import load_start_cache
from quadsmash.rl.supermimic_timebase_v2 import Timebase, bptt_return
from scripts.evaluate_supermimic_fresh_wm_panel import digest


def summarized(rows):
    if not rows:
        return dict(cases=0)
    true=torch.tensor([x['native_discounted_delta'] for x in rows],dtype=torch.float64)
    pred=torch.tensor([x['predicted_delta'] for x in rows],dtype=torch.float64)
    material=true.abs()>1.
    active=pred.ne(0)
    def match(mask):
        return float((true[mask].sign()==pred[mask].sign()).double().mean()) if mask.any() else None
    return dict(cases=len(rows),nonzero_predictions=int(active.sum()),native_mean=float(true.mean()),
                predicted_mean=float(pred.mean()),reward_prefix_mean=sum(x['reward_delta'] for x in rows)/len(rows),
                tail_value_mean=sum(x['bootstrap_delta'] for x in rows)/len(rows),
                same_sign=match(true.ne(0)),material_native_cases=int(material.sum()),same_sign_material=match(material),
                positive_native=int((true>0).sum()),positive_predicted=int((pred>0).sum()))


def annotate_landings(report, paired, dt, horizon):
    """Bind native contact/release evidence to each previously computed predicted delta."""
    baseline_hash,final_hash=paired[1]['landing_diagnostics']['raw_sha256']
    middle_hash,overlap_hash=paired[2]['landing_diagnostics']['raw_sha256']
    if final_hash!=overlap_hash or paired[2]['native_sources']!=report['fixed_sources'][1:]:
        raise ValueError('Different Recovery comparison')
    raw_hashes=[baseline_hash,middle_hash,final_hash]
    native=[]
    for source,expected in zip(report['fixed_sources'],raw_hashes,strict=True):
        path=Path(source['path'])/'episodes.jsonl'
        if digest(path)!=expected:
            raise ValueError('Native landing archive changed')
        with path.open() as f:
            native.append({r['case_id']:r for r in map(json.loads,f)})
    for name,(left,right) in zip(('R0_R8','R8_R16','R0_R16'),((0,1),(1,2),(0,2)),strict=True):
        matched=[]
        duration=[]
        for row in report['rows'][name]:
            a,b=(native[i][row['case_id']] for i in (left,right))
            if not row['qualified_both']:
                continue
            releases=[case['release_time_s'] for case in (a,b)]
            if any(t is None for t in releases):
                continue
            points=[torch.tensor(case['first_string_contact_point_m']) for case in (a,b)]
            position_error=(points[0]-points[1]).norm().item()
            row['precontact_native_evidence']=dict(release_times=releases,first_string_point_distance_m=position_error,
                start_precedes_both_releases=(row['start_tick']*dt <= min(releases)))
            events=[r['contact_flight_recovery'] for r in (a,b)]
            times=[e.get('contact_flight_landing_s') for e in events]
            if row['held'] and all(e.get('qualified_contact_landing') and t is not None for e,t in zip(events,times)):
                duration.append(max(0.,(row['start_tick']+horizon)*dt-max(times)))
            if (row['held'] and row['start_tick']*dt <= min(releases) and abs(releases[0]-releases[1])<=1.e-6
                    and position_error<=1.e-4):
                matched.append(row)
        report['scores'][name]['held_matched_release_and_point']=summarized(matched)
        report['scores'][name]['native_landing_in_horizon']=dict(available=len(duration),covered=sum(t>0 for t in duration),
            post_landing_seconds_median=float(torch.tensor(duration).median()) if duration else None)
    report['native_raw_sha256']=raw_hashes


@torch.no_grad()
def main():
    cli=argparse.ArgumentParser(description=__doc__)
    for k in ('r8','r16','paired-r8','paired-r16','paired-r8-to-r16','archive','safety-mapping','replay','output'):
        cli.add_argument('--'+k,type=Path,required=True)
    cli.add_argument('--device',default='cuda:0')
    cli.add_argument('--diagnostic-critic',type=Path)
    a=cli.parse_args()
    read=lambda f:json.loads(f.read_text())
    torch.set_num_threads(3)
    ck=torch.load(a.r16,mmap=True,weights_only=False,map_location='cpu')
    eight=torch.load(a.r8,mmap=True,weights_only=False,map_location='cpu')
    paired=[read(f) for f in (a.paired_r8,a.paired_r16)]
    if (not all(x['identical_initial_states'] and x['paired_case_identity_verified'] and x['case_count']==128 for x in paired)
            or paired[0]['native_sources'][0]!=paired[1]['native_sources'][0]):
        raise ValueError('Native panels do not share the fixed baseline')
    sources=[paired[0]['native_sources'][0],paired[0]['native_sources'][1],paired[1]['native_sources'][1]]
    meta=ck['metadata']['physical_imagination']
    paths=meta['checkpoint_paths']
    actor,adapter,template,launch,manifest=load_physical_loop(archive=a.archive,safety_mapping=a.safety_mapping,
        physical_checkpoint=paths['physical'],ball_checkpoint=paths['ball'],task_checkpoint=paths['task'],
        action_ball_checkpoint=paths.get('action_ball'),execution_root=meta['g07_execution_root'],device=a.device)
    weights=[{k:v.detach().cpu().clone() for k,v in actor.state_dict().items()},eight['actor'],ck['actor']]
    if manifest['actor_state_sha256']!=sources[0]['actor_sha256'] or adapter.manifest['checkpoint_sha256']!=meta['checkpoint_sha256']:
        raise ValueError('Initial archive or model dependencies changed')
    fit=read(a.replay/'fit.json')
    world=torch.load(a.replay/'online_world.pt',map_location='cpu',mmap=True,weights_only=True)
    if tensor_digest(world.items())!=fit['online_world_sha256'] or fit['stage_checkpoint_sha256']!=digest(a.r16):
        raise ValueError('Different R16 replay')
    adapter.load_state_dict(world,strict=True)
    adapter.set_contact_bptt_mode('detached')
    episodes=[]
    start_hash=None
    for src,w in zip(sources,weights,strict=True):
        if tensor_digest(w.items())!=src['actor_sha256'] or any(
                not torch.equal(v,weights[0][k]) for k,v in w.items() if k not in RECOVERY_NAMES):
            raise ValueError('Native policy changed outside Recovery')
        root=Path(src['path'])
        if (digest(root/'episodes_summary.json')!=src['episodes_sha256'] or digest(root/'metrics.json')!=src['metrics_sha256']
                or read(root/'metrics.json')['actor_state_sha256']!=src['actor_sha256']):
            raise ValueError('Native evaluation changed')
        actual=read(root/'initial_states.json')
        if start_hash is None:
            start_hash=actual
        if actual!=start_hash or src['seed']!=sources[0]['seed'] or read(root/'task_contract.json')!=manifest['task_contract']:
            raise ValueError('Different initial native cases or task')
        entries={r['env_id']:r for r in read(root/'episodes_summary.json')}
        if set(entries)!=set(range(128)):
            raise ValueError('Missing native case')
        episodes.append(entries)
    held=set(ck['config']['wm_holdout_lanes'])
    critic_state=ck['critic']
    if a.diagnostic_critic:
        diagnostic=torch.load(a.diagnostic_critic,weights_only=True,map_location='cpu')
        archive=ck['runtime_validation_archive']
        if (diagnostic['scope']!='landing_tail_order_diagnostic' or diagnostic['deployable']
                or diagnostic['base_critic_sha256']!=tensor_digest(ck['critic'].items())
                or diagnostic['archive_sha256']!=archive['sha256'] or digest(Path(archive['path']))!=archive['sha256']
                or diagnostic['actor_sha256']!=sources[-1]['actor_sha256']
                or set(diagnostic['held_lanes'])!=held):
            raise ValueError('Diagnostic critic uses different policy, data, or held lanes')
        critic_state=diagnostic['critic']
    starts=load_start_cache(a.archive,manifest)
    tb=Timebase(**read(Path(__file__).resolve().parents[1]/'configs/supermimic_task_horizon_v2.json'))
    assigned=set()
    comparisons={'R0_R8':[],'R8_R16':[],'R0_R16':[]}
    trainer=None
    for tick in (272,256,240):
        state=PhysicalImaginationStateV1.from_record(starts[tick],executor_template=template,launch=launch,device=a.device)
        possible=torch.where(~state.task_memory['rally']['had_contact'].reshape(-1).bool())[0].tolist()
        ids=[lane for lane in possible if lane not in assigned and all(v[lane]['episode_index']==int(state.clock_goal_context['episode_index'][lane]) for v in episodes)]
        if not ids:
            continue
        assigned.update(ids)
        if trainer is None:
            trainer=PhysicalHorizonTrainer(actor,adapter,state,tb,block_name='recovery',trainable_names=RECOVERY_NAMES)
            trainer.critic.load_state_dict(critic_state,strict=True)
        values=[]
        for w in weights:
            actor.load_state_dict(w,strict=True)
            out=trainer._rollout_objective(state)
            _,rw,dn,last,_=out['critic_data']
            lower=bptt_return(rw,dn,rw.new_zeros(last.shape[0],1),tb.gamma).flatten()
            values.append((out['ret'].flatten().cpu(),lower.cpu(),out['effective_weights'].cpu()))
        for name,(left,right) in zip(comparisons,((0,1),(1,2),(0,2)),strict=True):
            for lane in ids:
                a0,a1=episodes[left][lane],episodes[right][lane]
                if a0['case_id']!=a1['case_id'] or any(abs(x['native_return_gamma']-tb.gamma)>1e-10 for x in (a0,a1)):
                    raise ValueError('Changed discount or case assignment')
                # Discount each imagined suffix back to the original episode start.
                factor=tb.gamma**tick
                diff=float(values[right][0][lane]-values[left][0][lane])*factor
                reward=float(values[right][1][lane]-values[left][1][lane])*factor
                comparisons[name].append(dict(lane=lane,case_id=a0['case_id'],start_tick=tick,held=lane in held,
                    qualified_both=bool(a0['qualified_airborne_string_contact'] and a1['qualified_airborne_string_contact']),
                    recovery_flip=int(a1['recovery_success'])-int(a0['recovery_success']),
                    native_discounted_delta=a1['native_episode_discounted_return']-a0['native_episode_discounted_return'],
                    predicted_delta=diff,reward_delta=reward,bootstrap_delta=diff-reward,
                    imagined_recovery_gate=[bool(values[k][2][lane]) for k in (left,right)]))
    scores={name:{'all':summarized(rows),'held_qualified':summarized([r for r in rows if r['held'] and r['qualified_both']]),
                  'held_recovery_flips':summarized([r for r in rows if r['held'] and r['qualified_both'] and r['recovery_flip']])}
            for name,rows in comparisons.items()}
    report=dict(world_sha256=fit['online_world_sha256'],critic_sha256=tensor_digest(critic_state.items()),
        diagnostic_critic_sha256=digest(a.diagnostic_critic) if a.diagnostic_critic else None,
        fixed_sources=sources,optimizer_updates=0,excluded_no_precontact_start=sorted(set(range(128))-assigned),
        scores=scores,rows=comparisons,scope='Retrospective fixed-case episode deltas; imagination branches share a measured baseline pre-contact start and the stated critic. Updated policy prefix tensors were not archived. This is not a native counterfactual restart or a promotion receipt.')
    annotate_landings(report,[*paired,read(a.paired_r8_to_r16)],tb.control_dt_s,tb.imagination_steps)
    with a.output.open('x') as f:
        json.dump(report,f,indent=2,allow_nan=False)


if __name__=='__main__':
    main()
