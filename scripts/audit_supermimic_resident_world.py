"""Cross-replay resident native Approach trajectories with frozen world checkpoints."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import torch

from quadsmash.physics.full_racket_v4 import load_racket, string_strips
from quadsmash.rl.supermimic_archive_storage_v1 import read_archive_rows
from quadsmash.rl.supermimic_ball_sequence_v1 import ball_transition
from quadsmash.rl.supermimic_physical_dataset_v1 import decode_physical, encode_physical
from quadsmash.rl.supermimic_physical_loop_loading_v1 import load_physical_loop, tensor_digest
from quadsmash.rl.supermimic_release_calibration import racket_release_boundary, speed_bias_at_lead
from quadsmash.rl.supermimic_task_space_reference_v1 import TaskSpaceReferenceBankV1
from quadsmash.rl.supermimic_toi_racket_objective import string_boundary

from run_supermimic_approach_v2 import dump


def read(path):
    return json.loads(path.read_text())


def checkpoint(path):
    return torch.load(path, map_location='cpu', mmap=True, weights_only=False)


def category(row):
    if row['qualified_airborne_string_contact']:
        if row['outgoing_speed_mps'] < 4.5:
            return 'low'
        if row['ball_side_complete_except_recovery']:
            return 'marginal' if row['outgoing_speed_mps'] < 4.75 else 'success'
        return 'ball_path_or_rule'
    return 'frame' if row['native_contact']['frame_hit'] else 'other_contact'


def native_windows(directory, actor_hashes, lead, limit, scratch, task_contract):
    archive = directory / 'native_transitions'
    manifest = read(archive / 'manifest.json')
    if manifest['task_contract'] != task_contract:
        raise ValueError('Native archive and frozen world task contracts differ')
    records = [json.loads(line) for line in (directory / 'native_episodes.jsonl').read_text().splitlines()]
    candidates = {}
    for row in records:
        motion = row['native_contact']['first_approach_contact_motion']
        if (motion is None or row['numerical_failures'] or row['arm_identification_episode']
                or row['behavior_at_terminal']['actor_sha256'] not in actor_hashes):
            continue
        actor = row['behavior_at_terminal']['actor_sha256']
        lane, episode = row['env_id'], row['episode_index']
        elapsed = math.floor(motion['time_s'] / task_contract['control_dt_s'])
        if elapsed >= lead:
            candidates[lane, episode] = (row, actor, elapsed)

    by_lane = defaultdict(dict)
    for (lane, episode), entry in candidates.items():
        by_lane[lane][episode] = entry
    buffers = defaultdict(list)
    for chunk in manifest['chunks']:
        with read_archive_rows(archive, chunk, scratch=scratch) as rows:
            for record in rows:
                before, after = record['before'], record['after']
                for lane, episode in enumerate(before['episode_index'].tolist()):
                    candidate = by_lane[lane].get(episode)
                    if candidate is None:
                        continue
                    _, actor, elapsed = candidate
                    tick = int(before['elapsed_tick'][lane])
                    if (not elapsed - lead - 1 <= tick <= elapsed + 12
                            or record['actor_state_sha256'] != actor
                            or bool(record['arm_identification_episode'][lane])
                            or int(after['episode_index'][lane]) != episode):
                        continue
                    lane_slice = slice(lane, lane + 1)
                    buffers[lane, episode].append(dict(
                        tick=tick, native_tick=record['native_tick'], stream=record['stream_id'],
                        state=encode_physical({k: v[lane_slice] for k,v in before['physical'].items()})[0],
                        endpoint=encode_physical({k: v[lane_slice] for k,v in after['physical'].items()})[0],
                        targets=record['final_joint_targets'][lane].clone(),
                        previous=before['physical']['final_action'][lane].clone(),
                        feet=before['physical']['foot_contact'][lane].clone(),
                        present=before['physical']['ball_present'][lane].clone(),
                        elapsed=before['physical']['elapsed_s'][lane].clone(),
                        contact=bool(after['task']['had_contact'][lane] and not before['task']['had_contact'][lane]),
                    ))

    selected = defaultdict(list)
    retained = set()
    for identity, (native, actor, _) in sorted(candidates.items()):
        rows = buffers[identity]
        contact = next((index for index, row in enumerate(rows) if row['contact']), None)
        case_group = category(native)
        key = actor, case_group
        if contact is None or len(selected[key]) >= limit or (key,native['case_id']) in retained:
            continue
        window = rows[contact-lead:contact+9] if contact >= lead else []
        if (len(window) != lead+9 or len({r['stream'] for r in window}) != 1
                or any(b['tick'] != a['tick'] + 1 or b['native_tick'] != a['native_tick'] + 1
                       for a,b in zip(window, window[1:]))):
            continue
        retained.add((key,native['case_id']))
        selected[key].append((identity[0], native, window))
    return {digest: [case for (actor, _), cases in selected.items() if actor == digest for case in cases]
            for digest in actor_hashes}


def pose_at(adapter, state):
    body = decode_physical(state)
    return adapter.physical_model.geometry(
        body['joint_position'], body['joint_velocity'], body['root_position_w'],
        body['root_quaternion_w'], body['root_link_velocity_w'], body['root_angular_velocity_w'])


@torch.no_grad()
def replay(adapter, launch, cases, calibration, racket_config, separation, native_motion):
    lanes = torch.tensor([lane for lane,_,_ in cases])
    schedules = launch[lanes]
    steps = [{key: torch.stack([window[i][key] for _,_,window in cases])
              for key in ('state','endpoint','targets','previous','feet','present','elapsed')}
             for i in range(len(cases[0][2]))]
    state = steps[0]['state']
    present, feet = steps[0]['present'], steps[0]['feet'].float()
    prior = adapter.action_ball_model.impulse_prior
    strips = len(string_strips(load_racket(racket_config)))
    normal, offset = string_boundary(racket_config, state)
    racket = adapter.physical_model.geometry.ids['Racket']
    first = torch.full((len(lanes),), -1)
    parts = first.clone()
    release_tick = first.clone()
    release = state.new_full((len(lanes),6), torch.nan)
    errors = []
    for tick, row in enumerate(steps):
        if native_motion:
            state = torch.cat((row['state'][:,:49], state[:,49:]), dim=-1)
            feet = row['feet'].float()
        predicted, native = pose_at(adapter,state), pose_at(adapter,row['state'])
        pose_error = (predicted['position'][:,racket]-native['position'][:,racket]).norm(dim=-1)
        normal_error = (predicted['rotation'][:,racket,:,0]-native['rotation'][:,racket,:,0]).norm(dim=-1)
        point = row['state'][:,49:52]
        def point_velocity(pose):
            lever = point-pose['position'][:,racket]
            return pose['velocity'][:,racket]+torch.linalg.cross(pose['angular_velocity'][:,racket],lever)
        velocity_error = (point_velocity(predicted)-point_velocity(native)).norm(dim=-1)
        state,feet,present,details = ball_transition(
            adapter.physical_model,adapter.ball_model,state,row['targets'],row['previous'],feet,present,
            row['elapsed'],schedules,action_model=adapter.action_ball_model,return_details=True,
            native_robot_endpoint=row['endpoint'][:,:49] if native_motion else None)
        impact = details['impulse_info']
        new = (first < 0) & impact['active'] & (impact['object_id'] == 0)
        first[new],parts[new] = tick,impact['racket_part'][new]
        ball = details['predicted_ball']
        separated = ((first >= 0) & ~torch.isfinite(release[:,0]) &
                     racket_release_boundary(ball,pose_at(adapter,state),racket,prior.preview,separation_m=separation))
        release[separated] = ball[separated]
        release_tick[separated] = tick
        margin = (impact['local'][:,1:] @ normal.T - offset).amin(dim=-1)
        errors.append((pose_error.tolist(),normal_error.tolist(),velocity_error.tolist(),
                       [float(margin[j]) if new[j] else None for j in range(len(lanes))]))
    bias = speed_bias_at_lead(first, calibration)
    results = []
    for j,(lane,native,window) in enumerate(cases):
        hit = int(first[j])
        speed = float(release[j,3:].norm())
        kind = None if hit < 0 else 'string' if 0 <= parts[j] < strips else 'frame_or_nonstring'
        results.append(dict(
            case_id=native['case_id'],lane=lane,episode=native['episode_index'],stream=window[0]['stream'],
            in_wm_holdout=None,first_native_tick=window[0]['native_tick'],
            native_kind='string' if native['native_contact']['string_hit'] else
            'frame_or_nonstring' if native['native_contact']['frame_hit'] else 'other_contact',
            native_group=category(native),native_speed_mps=native['outgoing_speed_mps'],
            predicted_kind=kind,predicted_contact_step=hit if hit>=0 else None,
            signed_string_margin_m=errors[hit][3][j] if hit>=0 else None,
            predicted_release_step=int(release_tick[j]) if release_tick[j]>=0 else None,
            predicted_release_velocity_mps=release[j,3:].tolist() if math.isfinite(speed) else None,
            predicted_speed_mps=speed if math.isfinite(speed) else None,
            corrected_speed_mps=speed-float(bias[j]) if math.isfinite(speed) else None,
            racket_center_error_m_by_tick=[p[j] for p,_,_,_ in errors],
            normal_error_by_tick=[n[j] for _,n,_,_ in errors],
            contact_point_speed_error_mps_by_tick=[v[j] for _,_,v,_ in errors],
        ))
    return results


def summarize(rows, lead):
    speeds = [r for r in rows if r['native_kind'] == r['predicted_kind'] == 'string'
              and r['native_speed_mps'] is not None and r['predicted_speed_mps'] is not None]
    low = [r for r in speeds if r['native_speed_mps'] < 4.5]
    return dict(cases=len(rows),contact_matches=sum(r['predicted_kind']==r['native_kind'] for r in rows),
                string_releases=len(speeds),
                racket_center_at_native_contact_mae_m=mean(r['racket_center_error_m_by_tick'][lead] for r in rows),
                string_release_mae_mps=mean(abs(r['predicted_speed_mps']-r['native_speed_mps']) for r in speeds) if speeds else None,
                corrected_string_release_mae_mps=mean(abs(r['corrected_speed_mps']-r['native_speed_mps']) for r in speeds) if speeds else None,
                measured_low=len(low),
                false_fast=sum(r['predicted_speed_mps']>=4.5 for r in low),
                corrected_false_fast=sum(r['corrected_speed_mps']>=4.5 for r in low))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('capture','a40','a48','a56','registration','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--lead', type=int, default=20)
    parser.add_argument('--per-category', type=int, default=5)
    args = parser.parse_args()
    if args.lead < 1 or args.per_category < 1:
        parser.error('Lead and cases per category must be positive')
    torch.set_num_threads(3)
    checkpoints = {name: checkpoint(getattr(args,name)) for name in ('a40','a48','a56')}
    actor_ids = {name:tensor_digest(data['actor'].items()) for name,data in checkpoints.items()}
    experiment = checkpoints['a40']
    contract = experiment['metadata']['task_contract']
    holdout = experiment['config']['wm_holdout_lanes']
    if (len(set(actor_ids.values())) != len(actor_ids)
            or any(data['metadata']['task_contract'] != contract
                   or data['config']['wm_holdout_lanes'] != holdout for data in checkpoints.values())):
        raise ValueError('Expected distinct Actors with the same task and holdout lanes')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    windows = native_windows(args.capture, {actor_ids['a40'],actor_ids['a48']},
                             args.lead,args.per_category,args.output.parent,contract)
    if not all(windows.values()):
        raise ValueError('Native archive is missing a policy-consistent contact cohort')
    print({name:len(windows[actor_ids[name]]) for name in ('a40','a48')},flush=True)
    info = experiment['metadata']['physical_imagination']
    paths = info['checkpoint_paths']
    argv = read(args.registration)['command']
    _,adapter,_,launch,_ = load_physical_loop(
        archive=argv[argv.index('--archive')+1],safety_mapping=argv[argv.index('--safety-mapping')+1],
        physical_checkpoint=paths['physical'],ball_checkpoint=paths['ball'],task_checkpoint=paths['task'],
        action_ball_checkpoint=paths['action_ball'],execution_root=info['g07_execution_root'],device='cpu')
    geometry = Path(info['g07_execution_root'])/'configs/full_racket_v4_r3.json'
    separation = read(Path('configs/shuttle_physics_v2.json'))['separation_hysteresis_m']
    identities = {}
    for name,data in checkpoints.items():
        groups = defaultdict(list)
        for key,value in data['world'].items():
            prefix = key.split('.')[0] if not key.startswith('physical_model.body.') else 'body'
            groups[prefix].append((key,value))
        with getattr(args,name).open('rb') as stream:
            sha = hashlib.file_digest(stream,'sha256').hexdigest()
        reference = TaskSpaceReferenceBankV1.from_payload(data['reference_bank']) if data['reference_bank'] else None
        identities[name] = dict(actor_sha256=actor_ids[name],checkpoint_sha256=sha,
                                world_updates=data['world_updates'],native_ticks=data['native_ticks'],
                                model_sha256={k:tensor_digest(v) for k,v in groups.items()},
                                reference_bank_sha256=reference.content_sha256() if reference else None,
                                reference_bank=reference.manifest if reference else None,
                                approach_speed_calibration=data['approach_speed_calibration'])
    output = dict(lead_ticks=args.lead,identities=identities,cells={})
    for model in ('a40','a56','a40_with_a56_body'):
        data = checkpoints['a40'] if model == 'a40_with_a56_body' else checkpoints[model]
        adapter.load_state_dict(data['world'],strict=True)
        if model == 'a40_with_a56_body':
            prefix = 'physical_model.body.'
            adapter.physical_model.body.load_state_dict({key.removeprefix(prefix): value
                for key,value in checkpoints['a56']['world'].items() if key.startswith(prefix)}, strict=True)
        adapter.eval()
        for behavior in ('a40','a48'):
            for native_motion in ((False,) if model == 'a40_with_a56_body' else (False, True)):
                label = f'{behavior}_behavior_{model}_world_' + ('native_motion' if native_motion else 'recorded_commands')
                results = replay(adapter,launch,windows[actor_ids[behavior]],data['approach_speed_calibration'],
                                 geometry,separation,native_motion)
                for result in results:
                    result['in_wm_holdout'] = result['lane'] in holdout
                output['cells'][label] = dict(summary=summarize(results,args.lead),cases=results)
                print(label,output['cells'][label]['summary'],flush=True)
                dump(args.output,output)


if __name__ == '__main__':
    main()
