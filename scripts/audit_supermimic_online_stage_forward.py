#!/usr/bin/env python3
"""Score the actual online WM snapshot on recorded native launches after a policy block."""
import argparse
import json
from pathlib import Path

import torch

from evaluate_supermimic_fresh_wm_panel import assess, check_panel, digest
from rescore_supermimic_fresh_wm_panel import body_forward_check
from quadsmash.rl.supermimic_physical_loop_loading_v1 import load_physical_loop, tensor_digest
from quadsmash.rl.supermimic_physical_world_v1 import PhysicalWorldV1
from quadsmash.rl.supermimic_wm_candidate_gate_v1 import compare_candidate_to_anchor


def event_metrics(metrics):
    return dict(event_aligned={str(offset):{str(h):groups[f'contact{offset:+d}'] for h,groups in metrics.items()
                                            if f'contact{offset:+d}' in groups} for offset in (-1,0,1)})


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('panel','capture','initial-score','checkpoint','archive','safety-mapping','output'):
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--replayed-world',type=Path,help='Offline optimizer replay directory, bound to the same original run checkpoint')
    p.add_argument('--flight-oracle',type=Path,help='Attribute native 64-step errors to incoming ball, impact transition and robot motion using measured windows')
    p.add_argument('--attribute-flight',action='store_true',help='Partition uniform longest-horizon velocity error by observed launch/contact history')
    p.add_argument('--compare-components',action='store_true',help='Replay the panel with only body or ball weights from the online stage')
    a = p.parse_args()
    torch.set_num_threads(3)
    read = lambda path: json.loads(path.read_text())
    panel, measured, initial = read(a.panel), read(a.capture/'metrics.json'), read(a.initial_score)
    native = a.capture/'native.pt'
    root = Path(__file__).resolve().parents[1]
    anchor_path, behavior_path = [Path(path) for path in panel['weights']]
    check_panel(panel,root/'configs/g07_full_racket_eval_v4.json',root,[anchor_path,behavior_path])
    if (digest(a.panel)!=initial['panel_sha256'] or measured['panel_sha256']!=initial['panel_sha256'] or
        digest(native)!=measured['data_sha256'] or initial['data_sha256']!=measured['data_sha256'] or
        measured['ticks']!=800 or measured['first_episodes']!=len(panel['cases']) or
        initial['weights']!=panel['weights'] or measured['weights']!=panel['weights'] or
        initial['transition']!='initial_components' or not initial['initial_physical_components']):
        raise ValueError('Expected a complete, identical native panel and its deployed initial transition')
    behavior = torch.load(behavior_path,weights_only=False,map_location='cpu',mmap=True)
    if tensor_digest(behavior['actor'].items())!=measured['actor_sha256']:
        raise ValueError('Panel behavior policy identity mismatch')
    stage = torch.load(a.checkpoint,weights_only=False,map_location='cpu',mmap=True)
    wm = stage['metadata']['physical_imagination']
    paths = wm['checkpoint_paths']
    _,adapter,_,_,manifest = load_physical_loop(archive=a.archive,physical_checkpoint=paths['physical'],
        ball_checkpoint=paths['ball'],task_checkpoint=paths['task'],action_ball_checkpoint=paths.get('action_ball'),
        safety_mapping=a.safety_mapping,execution_root=wm['g07_execution_root'],device='cpu')
    if (behavior['metadata']['task_contract']!=stage['metadata']['task_contract'] or
        manifest['task_contract']!=stage['metadata']['task_contract'] or
        wm['checkpoint_sha256']!=adapter.manifest['checkpoint_sha256'] or
        initial['body_candidate_sha256']!=digest(Path(paths['physical'])) or initial['ball_candidate_sha256']!=digest(Path(paths['ball'])) or
        tensor_digest(adapter.state_dict().items())!=initial['runtime_world_sha256']):
        raise ValueError('Online snapshot has a different task or starting WM')
    initial_components = {}
    if a.compare_components:
        for name in ('physical_model','ball_model','action_ball_model'):
            module = getattr(adapter,name)
            if module is not None:
                initial_components[name] = {k:v.clone() for k,v in module.state_dict().items()}
    stage_sha = digest(a.checkpoint)
    trained = stage['world']
    replay_fit = None
    if a.replayed_world:
        registered=panel.get('online_world_comparison')
        if registered is not None:
            key=str(a.replayed_world.resolve())
            artifacts={str(a.replayed_world.resolve()/name):digest(a.replayed_world/name) for name in ('fit.json','online_world.pt')}
            if registered.get(key)!=artifacts:
                raise ValueError('Requested WM replay was not registered before native capture')
        replay_fit = read(a.replayed_world/'fit.json')
        trained = torch.load(a.replayed_world/'online_world.pt',weights_only=True,map_location='cpu',mmap=True)
        if (replay_fit['stage_checkpoint_sha256']!=stage_sha or replay_fit['updates']!=stage['world_updates'] or
                replay_fit['initial_world_sha256']!=initial['runtime_world_sha256'] or
                replay_fit['original_online_world_sha256']!=tensor_digest(stage['world'].items()) or
                replay_fit['online_world_sha256']!=tensor_digest(trained.items())):
            raise ValueError('Offline fit does not match this original online stage')
    adapter.load_state_dict(trained,strict=True)
    adapter.eval()
    anchor = PhysicalWorldV1().eval()
    anchor.load_state_dict(torch.load(anchor_path,weights_only=True,map_location='cpu')['model'])
    data = torch.load(native,weights_only=True,map_location='cpu',mmap=True)
    score = assess(data,panel['cases'],anchor,adapter,panel['horizons'],'cpu',flight_attribution=a.attribute_flight)
    if score['windows']!=measured['windows'] or score['windows']!=initial['windows']:
        raise ValueError('Different native windows')
    baseline = score['metrics']['anchor']
    current = score['metrics']['candidate']
    before = {int(h):v for h,v in initial['metrics']['candidate'].items()}
    for h in baseline:
        for group, row in baseline[h].items():
            for k,v in row.items():
                if abs(v-initial['metrics']['anchor'][str(h)][group][k])>1.e-5:
                    raise ValueError('Anchor does not reproduce the initial reference scores')
    names=panel.get('prospective_body_comparison',{}).get('uniform_no_regression',['body_rmse','root_angular_mae'])
    from scripts.supermimic_online_forward_guard import compare_forward
    report=dict(head_stage_checkpoint_sha256=stage_sha,stage_actor_sha256=tensor_digest(stage['actor'].items()),
        stage_world_sha256=tensor_digest(adapter.state_dict().items()),initial_world_sha256=initial['runtime_world_sha256'],
        native_panel_sha256=digest(a.panel),native_capture_sha256=measured['data_sha256'],initial_score_sha256=digest(a.initial_score),
        native_behavior_actor_sha256=measured['actor_sha256'],horizons=panel['horizons'],world_updates=stage['world_updates'],
        forward_guard_vs_formal_anchor=compare_forward(baseline,current,panel['horizons'],require_contact_improvement=True),
        forward_guard_vs_block_initial=compare_forward(before,current,panel['horizons']),
        uniform_vs_formal_anchor=body_forward_check(score['metrics'],panel['horizons'],names),
        contact_vs_formal_anchor=compare_candidate_to_anchor(event_metrics(baseline),event_metrics(current)),
        uniform_vs_block_initial=body_forward_check(dict(anchor=before,candidate=current),panel['horizons'],names),
        contact_vs_block_initial=compare_candidate_to_anchor(event_metrics(before),event_metrics(current)),
        metrics=current,scope='Offline monitoring on the same native launch panel with recorded executed targets. The online weights were generated after the initial panel declaration.',
        formal_promotion=False)
    if a.attribute_flight: report.update(uniform_horizon_attribution=score['uniform_horizon_attribution'],uniform_window_errors=score['uniform_window_errors'])
    if replay_fit is not None:
        report.update(replayed_world_sha256=report.pop('stage_world_sha256'),
                      stage_world_sha256=replay_fit['original_online_world_sha256'],
                      replay_fit_sha256=digest(a.replayed_world/'fit.json'),
                      learning_rate=replay_fit['learning_rate'],
                      scope='Offline training replay candidate evaluated on identical native launches and recorded targets.')
    if a.flight_oracle:
        from quadsmash.rl.supermimic_ball_sequence_v1 import ball_transition
        attribution=read(a.flight_oracle)
        if attribution['source_sha256']!=measured['data_sha256']:raise ValueError('Different native innovation data')
        records={(r['start'],r['lane']):r for r in attribution['windows']}
        uniform={(w['tick'],w['lane']) for w in score['windows'] if w['group']=='uniform'}
        if set(records)!=uniform:raise ValueError('Incomplete uniform innovation windows')
        peaks=[]
        for w in score['windows']:
            r=records.get((w['tick'],w['lane'])) if w['group']=='uniform' else None
            peaks.append(r['peak_tick']-r['start'] if r and r['peak_mps']>=.1 else -1000)
        peaks=torch.tensor(peaks)
        def summarize(rows):
            sums={}
            for r in rows:
                label=records[(r['tick'],r['lane'])]
                group=sums.setdefault(label['history'],{}).setdefault(label['closest_geometry'],dict(count=0,squared_error=0.))
                group['count']+=1;group['squared_error']+=r['squared_error']
            return sums
        report['flight_oracle']={}
        for mode in ('predicted','native_robot','native_ball_before_peak','native_ball_after_peak','native_robot_and_ball_before_peak'):
            step=0
            def transition(inputs,truth,patch):
                nonlocal step
                state,feet,present=ball_transition(adapter.physical_model,adapter.ball_model,*inputs,action_model=adapter.action_ball_model)
                if mode in ('native_robot','native_robot_and_ball_before_peak'):
                    state=torch.cat((truth[:,:49],state[:,49:]),-1)
                if 'ball_' in mode:
                    after='after' in mode
                    mask=peaks==step+(0 if after else 1)
                    state=torch.cat((state[:,:49],torch.where(mask[:,None],truth[:,49:],state[:,49:])),dim=-1)
                step+=1
                return state,feet,present,patch
            run=assess(data,panel['cases'],None,adapter,panel['horizons'],'cpu',predictors=('candidate',),
                       contact_step=transition,flight_attribution=True)
            values=run['metrics']['candidate'][64]['uniform']
            if mode=='predicted' and values!=current[64]['uniform']:raise ValueError('Flight attribution changed the reference rollout')
            report['flight_oracle'][mode]=dict(uniform=values,groups=summarize(run['uniform_window_errors']['candidate']))
        report['flight_oracle_source_sha256']=digest(a.flight_oracle)
        report['flight_oracle_scope']='Privileged interventions on recorded trajectories for error attribution only. Control-boundary resets include all six native ball coordinates.'
    if a.compare_components:
        report['component_comparisons'] = {}
        for group, restored in [('online_body_only',('ball_model','action_ball_model')),
                                ('online_ball_only',('physical_model',))]:
            adapter.load_state_dict(trained,strict=True)
            for name in restored:
                if name in initial_components:
                    getattr(adapter,name).load_state_dict(initial_components[name],strict=True)
            comparison = assess(data,panel['cases'],anchor,adapter,panel['horizons'],'cpu')
            if comparison['windows'] != score['windows']:
                raise ValueError('The component comparison uses different windows')
            report['component_comparisons'][group] = comparison['metrics']['candidate']
    with a.output.open('x') as f:
        json.dump(report,f,indent=2,allow_nan=False)
        f.write('\n')


if __name__=='__main__':main()
