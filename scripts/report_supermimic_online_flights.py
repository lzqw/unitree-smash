"""Compare observed completed flights with the same TRAIN128 cases before episode scoring."""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from quadsmash.rl.supermimic_archive_storage_v1 import read_archive_rows
from scripts.compare_supermimic_fixed_stages import frame_before_string, load_run


def release_speed(shot):
    velocity = shot['outgoing_velocity_mps']
    if shot['released'] and velocity is not None and all(v is not None and math.isfinite(v) for v in velocity):
        return math.hypot(*velocity)
    return None


def flight_flags(shot):
    velocity = shot['outgoing_velocity_mps']
    speed = release_speed(shot)
    contact = shot['legal_initial_contact'] and shot['string_hit'] and shot['airborne_string_hit']
    scoring_cross = contact and speed is not None and speed >= 5. and velocity[0]>0 and shot['crossed_net']
    return dict(had_contact=shot['had_contact'],frame_hit=shot['frame_hit'],qualified_contact=contact,
                scoring_cross=scoring_cross,scoring_target=scoring_cross and shot['landed_target'],ground_seen=shot['ground_seen'],airborne_string_hit=shot['airborne_string_hit'], crossed_net=shot['crossed_net'],
                shuttle_net_contact=shot['shuttle_net_contact'], landed_target=shot['landed_target'],
                released_at_least_5_mps=speed is not None and speed >= 5.)


def target_evidence(shot, flags):
    speed = release_speed(shot)
    failed = []
    if not flags['qualified_contact']:
        failed.append('qualified_contact')
    if speed is None:
        failed.append('measured_release')
    else:
        if not flags['released_at_least_5_mps']:
            failed.append('release_speed')
        if shot['outgoing_velocity_mps'][0] <= 0:
            failed.append('forward_release')
    for flag in ('crossed_net', 'landed_target'):
        if not flags[flag]:
            failed.append(flag)
    return dict(release_speed_mps=speed, failed_conditions=failed)



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
    frame_first = {}
    online_target = {}
    physics_dt = manifest['task_contract']['physics_dt_s']
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
                    first_string = after.get('auxiliary', {}).get('environment', {}).get('first_string_time')
                    if first_string is not None:
                        shot['first_string_time_s'] = float(first_string[lane])
                    shot['contact_time_s'] = float(b['contact_time'][lane])
                    frame_first[lane] = frame_before_string(shot, physics_dt)
                    finished[lane] = flight_flags(shot)
                    online_target[lane] = target_evidence(shot, finished[lane])
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
    before_frame = {i: frame_before_string(initial[i], physics_dt) for i in paired}
    first_contact_changes = dict(
        paired_cases=len(paired),
        unknown_initial_frame_order=[cases[i] for i in paired if before_frame[i] is None],
        unknown_online_frame_order=[cases[i] for i in paired if frame_first[i] is None],
        qualified_to_frame_first=[cases[i] for i,(a,b) in paired.items()
                                  if a['qualified_contact'] and not b['qualified_contact'] and frame_first[i]],
        frame_first_to_qualified=[cases[i] for i,(a,b) in paired.items()
                                  if not a['qualified_contact'] and b['qualified_contact'] and before_frame[i]])
    target_flips = []
    for lane, (before, after) in paired.items():
        if before['scoring_target'] != after['scoring_target']:
            target_flips.append(dict(case=cases[lane], lane=lane, gained=after['scoring_target'],
                baseline=target_evidence(initial[lane], before), online=online_target[lane],
                online_target_reward=float(native_rewards[lane, 2])))
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
                release_policy_versions=dict(Counter(releases[i] for i in paired if i in releases)),metrics=flags,first_contact_changes=first_contact_changes,
                target_score_flips=target_flips,landing_by_airborne_string_hit=contact_groups,shot_reward_totals=shooting_reward)


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
            for key, cases in result.get('first_contact_changes', {}).items():
                if isinstance(cases, list):
                    writer.add_scalar('first_contact/'+key,len(cases),step)
            if 'target_score_flips' in result:
                losses = [row for row in result['target_score_flips'] if not row['gained']]
                writer.add_scalar('target_score/sole_release_speed_losses',
                    sum(row['online']['failed_conditions'] == ['release_speed'] for row in losses), step)
            for metric,totals in result['shot_reward_totals'].items():
                delta=totals['online_recorded']-totals['initial_from_native_latches']
                writer.add_scalar('shot_reward_difference_per_case/'+metric,delta/n,step)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--arm-dir',type=Path,required=True)
    cli.add_argument('--tensorboard',type=Path)
    cli.add_argument('--follow-pid',type=int,help='Wait for the first 512 native ticks and finish by tick 800')
    cli.add_argument('--output',type=Path,help='Save the paired first-flight case report')
    args=cli.parse_args()
    def emit():
        result=report(args.arm_dir)
        rendered=json.dumps(result,indent=2,allow_nan=False)
        if args.output:
            temp=args.output.with_suffix('.tmp')
            temp.write_text(rendered+'\n')
            temp.replace(args.output)
        if args.tensorboard:
            publish(result,args.tensorboard)
        print(rendered,flush=True)
        return result
    if args.follow_pid is None:
        emit()
    else:
        import psutil
        import time
        owner=psutil.Process(args.follow_pid)
        command=owner.cmdline()
        if ('scripts/train_supermimic_physical_block.py' not in command or
                Path(command[command.index('--output')+1]).resolve() != args.arm_dir.resolve()/'train'):
            raise ValueError('Training process belongs to a different arm')
        manifest=args.arm_dir/'train/native_transitions/manifest.json'
        for milestone in (512,800):
            while not manifest.is_file() or json.loads(manifest.read_text())['control_ticks'] < milestone:
                if not owner.is_running() or owner.status()==psutil.STATUS_ZOMBIE:
                    raise SystemExit('Training ended before the first-flight panel was complete')
                time.sleep(20)
            result=emit()
            if result['online_resolved']==result['cases'] or result['manifest_env_steps']>=800*result['cases']:
                break
