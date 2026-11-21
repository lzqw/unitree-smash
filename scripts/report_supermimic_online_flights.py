"""Compare observed completed flights with the same TRAIN128 cases before episode scoring."""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from quadsmash.rl.supermimic_archive_storage_v1 import read_archive_rows
from scripts.compare_supermimic_fixed_stages import load_run


def flight_flags(shot):
    velocity = shot['outgoing_velocity_mps']
    speed = math.hypot(*velocity) if shot['released'] and all(math.isfinite(v) for v in velocity) else None
    contact = shot['legal_initial_contact'] and shot['string_hit'] and shot['airborne_string_hit']
    scoring_cross = contact and speed is not None and speed >= 5. and velocity[0]>0 and shot['crossed_net']
    return dict(had_contact=shot['had_contact'],frame_hit=shot['frame_hit'],qualified_contact=contact,
                scoring_cross=scoring_cross,scoring_target=scoring_cross and shot['landed_target'],ground_seen=shot['ground_seen'],airborne_string_hit=shot['airborne_string_hit'], crossed_net=shot['crossed_net'],
                shuttle_net_contact=shot['shuttle_net_contact'], landed_target=shot['landed_target'],
                released_at_least_5_mps=speed is not None and speed >= 5.)


def report(arm):
    archive = arm/'train/native_transitions'
    manifest = json.loads((archive/'manifest.json').read_text())
    reg = json.loads((arm/'registration.json').read_text())
    baseline = Path(reg['baseline_path']).parent
    _, identity, actor = load_run(baseline)
    if (actor != reg['initial']['actor_sha256'] or identity['task'] != manifest['task_contract']
            or identity['execution'] != manifest['policy_execution'] or identity['seed'] != manifest['seed']):
        raise ValueError('Different TRAIN128 baseline actor, task, reset seed or execution')
    suite = identity['task']['suite']
    source = (Path(__file__).resolve().parents[1]/suite['path']).read_bytes()
    cases = [c['case_id'] for c in json.loads(source)['cases']]
    if (hashlib.sha256(source).hexdigest() != suite['sha256'] or
            sorted((case,i) for i,case in enumerate(cases)) != identity['assignments']):
        raise ValueError('TRAIN128 case assignment differs from the baseline')
    initial = {}
    for line in (baseline/'episodes.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row['episode_index'] == 0:
            lane = row['env_id']
            if lane in initial or cases[lane] != row['case_id']:
                raise ValueError('Ambiguous baseline first flight')
            initial[lane] = row
    if len(initial) != len(cases):
        raise ValueError('Missing baseline first episodes')
    import torch
    releases = {}
    finished = {}
    native_rewards = None
    for chunk in manifest['chunks']:
        with read_archive_rows(archive,chunk,scratch=arm) as rows:
            for row in rows:
                before,after = row['before'],row['after']
                a,b = before['task'],after['task']
                reward = row['reward_components']
                tick_rewards = torch.stack([reward[k].reshape(-1) for k in ('qualified_contact','cross','target')],1)
                tick_rewards = tick_rewards*(before['episode_index']==0).reshape(-1,1)
                if native_rewards is None:
                    native_rewards = tick_rewards.clone()
                else:
                    native_rewards += tick_rewards
                same = (before['episode_index']==0) & (after['episode_index']==0)
                for lane in (same & ~a['released'] & b['released']).nonzero().flatten().tolist():
                    releases.setdefault(lane,row['policy_version'])
                ended = same & b['launched'] & ~b['in_play'] & (a['in_play'] | ~a['launched'])
                for lane in ended.nonzero().flatten().tolist():
                    if lane in finished:
                        raise ValueError('Multiple finished flights recorded for one case')
                    shot = {k: bool(b[k][lane]) for k in ('had_contact','frame_hit','legal_initial_contact','string_hit','ground_seen','released','airborne_string_hit','crossed_net',
                                                          'shuttle_net_contact','landed_target')}
                    shot['outgoing_velocity_mps'] = b['outgoing_velocity'][lane].tolist()
                    finished[lane] = flight_flags(shot)
        if len(finished)==len(cases):
            break
    paired = {i:(flight_flags(initial[i]),finished[i]) for i in sorted(finished)
              if initial[i]['launched'] and not initial[i]['in_play']}
    flags = {}
    for key in next(iter(finished.values()),{}):
        improved = [cases[i] for i,(a,b) in paired.items() if b[key] and not a[key]]
        lost = [cases[i] for i,(a,b) in paired.items() if a[key] and not b[key]]
        flags[key] = dict(initial=sum(a[key] for a,b in paired.values()),online=sum(b[key] for a,b in paired.values()),
                          false_to_true=improved,true_to_false=lost)
    contact_groups = {}
    for had,has in ((True,True),(True,False),(False,True),(False,False)):
        group = [(a,b) for a,b in paired.values() if (a['airborne_string_hit'],b['airborne_string_hit'])==(had,has)]
        contact_groups[f'{int(had)}_to_{int(has)}'] = dict(cases=len(group),
            initial_landed_target=sum(a['landed_target'] for a,b in group),
            online_landed_target=sum(b['landed_target'] for a,b in group),
            online_frame_hit=sum(b['frame_hit'] for a,b in group),online_no_contact=sum(not b['had_contact'] for a,b in group))
    shooting_reward = {}
    for index,(metric,weight) in enumerate((('qualified_contact',20.),('scoring_cross',5.),('scoring_target',15.))):
        observed = float(native_rewards[list(paired),index].sum()) if paired else 0.
        shooting_reward[metric] = dict(initial_from_native_latches=weight*sum(a[metric] for a,b in paired.values()),
                                      online_from_native_latches=weight*sum(b[metric] for a,b in paired.values()),
                                      online_recorded=observed)
    return dict(scope='First flight leaving play after launch, including faults before ground contact. Compare native latches to first-episode frozen terminal records on the same resolved cases. The online prefix spans behavior versions. Episode recovery, official ball-side outcome and final Return require complete scoring.',
                initial_actor_sha256=actor,seed=identity['seed'],suite_sha256=suite['sha256'],
                manifest_env_steps=manifest['transitions'],cases=len(cases),online_resolved=len(finished),
                paired_resolved_cases=len(paired),unresolved_online_cases=[c for i,c in enumerate(cases) if i not in finished],
                unresolved_baseline_cases=[cases[i] for i in finished if i not in paired],
                release_policy_versions=dict(Counter(releases[i] for i in paired if i in releases)),metrics=flags,landing_by_airborne_string_hit=contact_groups,shot_reward_totals=shooting_reward)


def publish(result, folder):
    from torch.utils.tensorboard import SummaryWriter
    with SummaryWriter(str(folder)) as writer:
        step=result['manifest_env_steps']
        writer.add_scalar('paired_resolved_cases',result['paired_resolved_cases'],step)
        for metric,values in result['metrics'].items():
            for key,count in values.items():
                writer.add_scalar(f'{metric}/{key}',len(count) if isinstance(count,list) else count,step)
        n=result['paired_resolved_cases']
        if n:
            for metric,totals in result['shot_reward_totals'].items():
                delta=totals['online_recorded']-totals['initial_from_native_latches']
                writer.add_scalar('shot_reward_difference_per_case/'+metric,delta/n,step)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--arm-dir',type=Path,required=True)
    cli.add_argument('--tensorboard',type=Path)
    args=cli.parse_args()
    result=report(args.arm_dir)
    if args.tensorboard:
        publish(result,args.tensorboard)
    print(json.dumps(result,indent=2,allow_nan=False))
