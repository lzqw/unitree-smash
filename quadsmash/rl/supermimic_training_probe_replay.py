"""Import completed paired interventions into the supervised WM replay only."""
import json
from pathlib import Path

import torch

from .supermimic_action_identification_dataset_v1 import _load_branch, _same_contract, _sha256
from .supermimic_native_pair_capture_v1 import BRANCHES, tree_digest


def training_windows(directory, task_contract, policy_execution, excluded_lanes=(), *, expected_schema=None):
    directory = Path(directory)
    status = json.loads((directory/'status.json').read_text())
    if status['status'] != 'COLLECTION_FINISHED':
        raise ValueError('Incomplete action-identification collection')
    startup = json.loads((directory/'startup.json').read_text())
    specs = {Path(item['path']).stem:item['spec'] for item in startup['specs']}
    groups = {}
    for item in status['completed']:
        name, branch = item['spec'], item['branch']
        if branch not in BRANCHES or name not in specs or branch in groups.setdefault(name, {}):
            raise ValueError('Unexpected or repeated native branch')
        path = directory/name/branch/'native_probe.pt'
        if item['exit_code'] != 0 or _sha256(path) != item['probe_sha256']:
            raise ValueError('Native probe hash or exit code mismatch')
        groups[name][branch] = item['probe_sha256']
    if set(groups) != set(specs) or not groups or len(status['completed']) != status['total']:
        raise ValueError('Missing native branches')
    staged = []
    included = {}
    reserved = set(excluded_lanes)
    for name, hashes in sorted(groups.items()):
        if set(hashes) != set(BRANCHES):
            raise ValueError('Every training intervention must have two nominal and two signed runs')
        runs = {b:_load_branch(directory/name,b) for b in BRANCHES}
        _same_contract(runs)
        base = runs['nominal_a']
        spec = specs[name]
        if (base['spec'] != spec or not spec.get('capture_wm_window')
                or spec.get('data_role') != 'training_action_identification'
                or base['task_contract'] != task_contract or base['policy_execution'] != policy_execution):
            raise ValueError('Training probe has incompatible physics, execution, or split')
        if len(set(spec['lanes'])) != len(spec['lanes']):
            raise ValueError('Duplicate lane in signed native training windows')
        indices = [i for i,lane in enumerate(spec['lanes']) if lane not in reserved]
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
            if b in ('minus','plus'):
                tensors = window['tensors']
                if any(v.shape[:2] != (16,len(spec['lanes'])) for v in tensors.values()):
                    raise ValueError('Native window must cover the full 16-step signed sequence')
                values = {k:v[:,indices] for k,v in tensors.items()}
                if ((values['terminated'] | values['timeout']).any() or
                        not (values['episode'] == values['episode'][0]).all()):
                    raise ValueError('Signed native training window crosses reset')
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
                signed_streams=2*len(staged), world_window_steps=16)


def import_training_collection(replay, directory, task_contract, policy_execution, excluded_lanes=()):
    pairs, receipt = training_windows(directory, task_contract, policy_execution, excluded_lanes,
                                      expected_schema=replay.schema.manifest())
    streams = [replay.training_probe_stream(lanes, signed[b])
               for lanes, signed in pairs for b in ('minus', 'plus')]
    replay.training_probes.extend(streams)
    replay._ordered_cache = None
    return receipt
