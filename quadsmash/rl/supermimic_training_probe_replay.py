"""Import completed paired interventions into the supervised WM replay only."""
import json
from pathlib import Path

import torch

from .supermimic_action_identification_dataset_v1 import _load_branch, _same_contract, _sha256
from .supermimic_native_pair_capture_v1 import BRANCHES, TRAINING_WINDOW_STEPS, probe_branches, tree_digest



def verified_terminal_records(run):
    """Resolve only same-branch, same-actor terminal rows attached to the exact start."""
    records = run.get('terminal_labels', {})
    spec = run['spec']
    probe = tree_digest(spec)
    prefix = run['prefix_sha256']
    branch = run['branch']
    actor = run.get('candidate_actor_sha256') or tree_digest(run['actor_state'])
    stream = tree_digest((probe, prefix, run['seed'], branch))
    result = {}
    for lane, row in records.items():
        expected = dict(probe=probe, prefix=prefix, branch=branch, actor=actor, stream=stream,
                        env_id=lane, episode_index=int(run['raw']['start']['before']['episode_index'][lane]))
        if lane not in spec['lanes'] or any(row.get(k) != v for k, v in expected.items()):
            raise ValueError('Terminal label belongs to a different native probe episode')
        if row['scoring_schema'] != 'approach_ball_only_v1':
            raise ValueError('Unverified ball-side terminal scoring')
        if not row['approach_ball_terminal']['ball_terminal'] or row['numerical_failures']:
            continue
        result[lane] = row
    return result


def terminal_response_group(row):
    if row is None or row['approach_ball_terminal']['numerical_abort']:
        return 3
    speed = row['outgoing_speed_mps']
    if row['qualified_airborne_string_contact'] and speed is not None and 'speed_threshold' in row['rejection_reasons']:
        return 0
    if row['ball_side_complete_except_recovery'] and speed is not None:
        return 1 if speed < 4.75 else 2
    return 3

def training_windows(directory, task_contract, policy_execution, excluded_lanes=(), *, expected_schema=None,
                     directed_validation_lanes=()):
    directory = Path(directory)
    status = json.loads((directory/'status.json').read_text())
    if status['status'] != 'COLLECTION_FINISHED':
        raise ValueError('Incomplete action-identification collection')
    startup = json.loads((directory/'startup.json').read_text())
    specs = {Path(item['path']).stem:item['spec'] for item in startup['specs']}
    groups = {}
    for item in status['completed']:
        name, branch = item['spec'], item['branch']
        if name not in specs or branch not in probe_branches(specs[name]) or branch in groups.setdefault(name, {}):
            raise ValueError('Unexpected or repeated native branch')
        path = directory/name/branch/'native_probe.pt'
        if item['exit_code'] != 0 or _sha256(path) != item['probe_sha256']:
            raise ValueError('Native probe hash or exit code mismatch')
        groups[name][branch] = item['probe_sha256']
    if set(groups) != set(specs) or not groups or len(status['completed']) != status['total']:
        raise ValueError('Missing native branches')
    staged = []
    included = {}
    collected_steps = {}
    reserved = set(excluded_lanes)
    for name, hashes in sorted(groups.items()):
        spec = specs[name]
        if set(hashes) != set(probe_branches(spec)):
            raise ValueError('Every training intervention must have two nominal and two signed runs')
        runs = {b:_load_branch(directory/name,b) for b in BRANCHES}
        _same_contract(runs)
        base = runs['nominal_a']
        expected_task, expected_execution = task_contract, policy_execution
        if spec.get('terminal_tick_cap'):
            from .supermimic_policy_execution_v1 import contract
            from .supermimic_terminal_scoring import approach_ball_scoring_protocol
            expected_task = dict(task_contract, approach_ball_scoring=approach_ball_scoring_protocol())
            expected_execution = contract(arm_preparation_seconds=.25)
        if (base['spec'] != spec or not spec.get('capture_wm_window')
                or spec.get('data_role') != 'training_action_identification'
                or base['task_contract'] != expected_task or base['policy_execution'] != expected_execution):
            raise ValueError('Training probe has incompatible physics, execution, or split')
        steps = spec['tick'] - spec.get('action_tick', spec['tick']) + 1
        if steps not in TRAINING_WINDOW_STEPS:
            raise ValueError('Unsupported signed native training window length')
        collected_steps[name] = steps
        if len(set(spec['lanes'])) != len(spec['lanes']):
            raise ValueError('Duplicate lane in signed native training windows')
        excluded = reserved | (set(directed_validation_lanes) if spec.get('terminal_tick_cap') else set())
        indices = [i for i,lane in enumerate(spec['lanes']) if lane not in excluded]
        if not indices:
            raise ValueError('No training lanes remain after applying the WM holdout')
        lanes = [spec['lanes'][i] for i in indices]
        included[name] = lanes
        actor_id = tree_digest(base['actor_state'])
        signed = {}
        for b,run in runs.items():
            if tree_digest(run['actor_state']) != actor_id:
                raise ValueError('Different actor weights within native pair')
            window = run['raw']['wm_window']
            if ((expected_schema is not None and window['schema'] != expected_schema)
                    or window['schema'] != base['raw']['wm_window']['schema'] or window['lanes'] != spec['lanes']):
                raise ValueError('Native probe task-memory schema or lane order differs')
            tensors = window['tensors']
            if any(v.shape[:2] != (steps, len(spec['lanes'])) for v in tensors.values()):
                raise ValueError('Native window must cover the full declared signed sequence')
            if b in ('minus','plus'):
                values = {k:v[:,indices] for k,v in tensors.items()}
                if ((values['terminated'] | values['timeout']).any() or
                        not (values['episode'] == values['episode'][0]).all()):
                    raise ValueError('Signed native training window crosses reset')
                if spec.get('terminal_tick_cap'):
                    terminals = verified_terminal_records(run)
                    codes = torch.tensor([terminal_response_group(terminals.get(lane)) for lane in lanes])
                    values['terminal_response_group'] = codes[None].expand(steps, -1)
                signed[b] = values
        first, repeat = (runs[b]['raw']['wm_window']['tensors'] for b in ('nominal_a', 'nominal_b'))
        for key in ('state', 'next_state', 'targets', 'previous_action', 'applied_joint_torque'):
            if not torch.equal(first[key], repeat[key]):
                raise ValueError(f'Nominal native windows differ in {key}')
        for b in ('minus', 'plus'):
            if not torch.equal(signed[b]['state'][0], first['state'][0, indices]):
                raise ValueError('Signed native windows lack a common initial state')
        staged.append((lanes, signed))
    return staged, dict(path=str(directory.resolve()), status_sha256=_sha256(directory/'status.json'),
                source_checkpoint_sha256=startup['source_sha256'], spec_branches=groups,
                included_lanes=included, excluded_lanes=sorted(reserved),
                measured_windows=sum(2*len(lanes) for lanes in included.values()),
                signed_streams=2*len(staged), world_window_steps=16, signed_window_steps=collected_steps,
                terminal_labeled_collection=any(spec.get('terminal_tick_cap') for spec in specs.values()),
                directed_validation_lanes=list(directed_validation_lanes))


def import_training_collection(replay, directory, task_contract, policy_execution, excluded_lanes=(), *,
                               original_lanes=None, directed_validation_lanes=()):
    """Map authenticated suite lane IDs to the actual replay layout at import time."""
    if original_lanes is None:
        lane_map = None
        reserved = set(excluded_lanes)
    else:
        if (len(original_lanes) != replay.lanes or len(set(original_lanes)) != replay.lanes
                or any(type(lane) is not int or not 0 <= lane < 128 for lane in original_lanes)):
            raise ValueError('Invalid original case IDs for the training panel')
        if set(original_lanes).intersection(excluded_lanes):
            raise ValueError('Training panel includes reserved cases')
        lane_map = {lane: index for index, lane in enumerate(original_lanes)}
        reserved = set(range(128)) - set(original_lanes)
    options = {'expected_schema': replay.schema.manifest()}
    if directed_validation_lanes:
        options['directed_validation_lanes'] = directed_validation_lanes
    pairs, receipt = training_windows(directory, task_contract, policy_execution, reserved, **options)
    streams = []
    signed_pairs = []
    for lanes, signed in pairs:
        local = lanes if lane_map is None else [lane_map[lane] for lane in lanes]
        for branch in ('minus', 'plus'):
            streams.append(replay.training_probe_stream(local, signed[branch]))
        signed_pairs.append((tuple(local), signed['minus'], signed['plus']))
    replay.training_probes.extend(streams)
    if hasattr(replay, 'training_probe_pairs'):
        replay.training_probe_pairs.extend(signed_pairs)
    replay._ordered_cache = None
    if lane_map is not None:
        receipt['original_lane_by_replay_lane'] = list(original_lanes)
    return receipt


def directed_start_events(directory, receipt):
    """Yield native starts with authenticated delayed outcomes for the signed collection."""
    from .supermimic_actor_start_bank_v1 import ball_start_outcome
    from .supermimic_physical_capture_v1 import tensor_tree_copy

    directory = Path(directory)
    for name, lanes in receipt['included_lanes'].items():
        for branch in ('nominal_a', 'minus', 'plus'):
            path = directory / name / branch
            if _sha256(path / 'native_probe.pt') != receipt['spec_branches'][name][branch]:
                raise ValueError('Directed start capture changed after probe verification')
            run = _load_branch(directory / name, branch)
            if not run['spec'].get('terminal_tick_cap'):
                continue
            labels = verified_terminal_records(run)
            with (path / 'episodes.jsonl').open() as source:
                first = {int(row['env_id']): row for row in map(json.loads, source)}
            for lane, label in labels.items():
                raw = first[lane]
                if raw['case_id'] != label['case_id'] or raw['episode_index'] != label['episode_index']:
                    raise ValueError('Terminal metadata and native event record identities differ')
                label['native_flight'] = {key: raw.get(key) for key in (
                    'shuttle_net_contact', 'crossed_net', 'landed_target', 'net_margin_m', 'first_landing_m')}
                label['robot_net_fault'] = bool(raw['robot_net_fault'])
                label['contact_time_s'] = raw.get('contact_time_s')
            start = tensor_tree_copy(run['raw']['start'], cpu=True)
            start['_directed_training_prefix'] = True
            allowed = {lane: label for lane, label in labels.items()
                       if lane in lanes and ball_start_outcome(label) is not None}
            if not allowed:
                continue
            mask = torch.zeros(len(start['before']['episode_index']), dtype=torch.bool)
            mask[list(allowed)] = True
            stream = int(next(iter(allowed.values()))['stream'][:12], 16)
            start['stream_id'] = stream
            entry = dict(start=start, stream=stream, policy_version=0, native_tick=run['spec']['action_tick'],
                         masks=dict(pre_contact=mask), ball_outcomes={
                             lane: (ball_start_outcome(label), None) for lane, label in allowed.items()})
            yield entry, allowed
