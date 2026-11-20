"""Import completed paired interventions into the supervised WM replay only."""
import json
from pathlib import Path

import torch

from .supermimic_action_identification_dataset_v1 import _load_branch, _same_contract, _sha256
from .supermimic_native_pair_capture_v1 import BRANCHES, tree_digest


def import_training_collection(replay, directory, task_contract, policy_execution, excluded_lanes=()):
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
    if not groups or len(status['completed']) != status['total']:
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
        indices = [i for i,lane in enumerate(spec['lanes']) if lane not in reserved]
        if not indices:
            raise ValueError('No training lanes remain after applying the WM holdout')
        lanes = [spec['lanes'][i] for i in indices]
        included[name] = lanes
        actor_id = tree_digest(base['actor_state'])
        for b,run in runs.items():
            if tree_digest(run['actor_state']) != actor_id:
                raise ValueError('Different actor weights within native pair')
            window = run['raw']['wm_window']
            if window['schema'] != replay.schema.manifest() or window['lanes'] != spec['lanes']:
                raise ValueError('Native probe task-memory schema or lane order differs')
            if b in ('minus','plus'):
                values = {k:v[:,indices] for k,v in window['tensors'].items()}
                staged.append(replay.training_probe_stream(lanes, values))
    replay.training_probes.extend(staged)
    replay._ordered_cache = None
    return dict(path=str(directory.resolve()), status_sha256=_sha256(directory/'status.json'),
                source_checkpoint_sha256=startup['source_sha256'], spec_branches=groups,
                included_lanes=included, excluded_lanes=sorted(reserved),
                measured_windows=sum(2*len(lanes) for lanes in included.values()),
                signed_streams=len(staged), world_window_steps=16)
