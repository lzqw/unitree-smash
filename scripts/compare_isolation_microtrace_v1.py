"""Compare paired native physics windows without inferring collision causality."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

STATE_FIELDS = ('joint_position', 'joint_velocity', 'root_link_position',
    'root_link_quaternion', 'root_com_linear_velocity', 'root_com_angular_velocity')
SUBMITTED_FIELDS = ('submitted_position_target', 'submitted_velocity_target', 'submitted_effort_target')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_window(folder, case):
    metadata_path = folder/'isolation_microtrace.json'
    path = folder/'isolation_microtrace.npz'
    metadata = json.loads(metadata_path.read_text())
    if (not metadata['complete_window'] or metadata['data_sha256'] != sha(path)
            or case not in metadata['lanes'] or metadata['action_changed'] or metadata['physics_changed']):
        raise ValueError('Require complete unchanged passive telemetry')
    lane = metadata['lanes'].index(case)
    count = metadata['last_frame']-metadata['first_frame']+1
    with np.load(path, allow_pickle=False) as z:
        frames = z['physics_frame'].copy()
        if not np.array_equal(frames,np.arange(metadata['first_frame'],metadata['last_frame']+1)):
            raise ValueError('Missing physical frames')
        values = {}
        for key in z.files:
            if key == 'physics_frame':
                continue
            value = z[key]
            if value.shape[0] != count or not np.isfinite(value).all():
                raise ValueError('Nonfinite or incomplete '+key)
            # Wrench-enable/frame flags are asset-global scalar flags; all
            # remaining channels explicitly have a recorded lane dimension.
            values[key] = value.copy() if value.ndim == 1 else value[:,lane].copy()
    if not np.all(values['episode_index']==values['episode_index'][0]):
        raise ValueError('Episode reset within physical comparison')
    return metadata, frames, values, {str(p.resolve()):sha(p) for p in [path,metadata_path]}


def compare_values(frames, before, after):
    if before.keys() != after.keys():
        raise ValueError('Different telemetry fields')
    result = {}
    for key in before:
        if before[key].shape != after[key].shape:
            raise ValueError('Different telemetry shape '+key)
        diff = abs(after[key].astype(float)-before[key].astype(float)).reshape(len(frames),-1)
        error = diff.max(1)
        changed = np.flatnonzero(error>0)
        first = int(changed[0]) if len(changed) else None
        result[key] = dict(first_difference_frame=None if first is None else int(frames[first]),
            maximum_absolute_difference=float(error.max()),
            first_difference_maximum=None if first is None else float(error[first]))
    candidates = [result[k]['first_difference_frame'] for k in STATE_FIELDS]
    candidates = [x for x in candidates if x is not None]
    first_state = min(candidates) if candidates else None
    prefix = None if first_state is None else frames <= first_state
    return dict(fields=result, first_physical_state_difference_frame=first_state,
        submitted_buffers_equal_through_first_state_difference=None if prefix is None else
            all(np.array_equal(before[k][prefix],after[k][prefix]) for k in SUBMITTED_FIELDS),
        cause_identified=False)


def compare(before, after, case):
    old_meta, frames, old, files = read_window(before,case)
    meta, other_frames, current, more = read_window(after,case); files.update(more)
    for key in ['physics_dt_s','lanes','first_frame','last_frame','joint_names','foot_names','origins_world','source_sha256']:
        if old_meta[key] != meta[key]:
            raise ValueError('Different telemetry context '+key)
    if not np.array_equal(frames,other_frames):
        raise ValueError('Different physics timeline')
    result = compare_values(frames,old,current)
    files[str(Path(__file__).resolve())]=sha(__file__)
    return dict(result,case=case,before=str(before.resolve()),after=str(after.resolve()),
        source_sha256=files,physics_dt_s=meta['physics_dt_s'],
        source_native_jobs_not_admitted_by_this_comparison=True,
        valid_expert_labels=False, strict_audit_tolerances_changed=False,
        interpretation='Same-frame differences do not establish causal ordering inside PhysX. Submitted effort is a control buffer, not realized implicit-PD torque.')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['before','after','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--case',type=int,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise ValueError('Preserve prior comparison')
    result=compare(a.before,a.after,a.case)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream:
        stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))
