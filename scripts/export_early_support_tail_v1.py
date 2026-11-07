"""Export a replay-verified fine recovery candidate for ground-start validation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quadsmash.rl.episode_trace_v4 import unpack_public


def export(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists(): raise ValueError('Fresh bundle required')
    read = lambda p: json.loads(Path(p).read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    result = read(source)
    hashes = dict(result['source_sha256'])
    hashes[str(source)] = sha(source)
    hashes[str(Path(__file__).resolve())] = sha(__file__)
    for name, expected in hashes.items():
        if sha(name) != expected: raise ValueError('Changed candidate source '+name)
    replay = result['independent_replay']
    if (not result['any_model_feasible'] or not result['replay_reproduced'] or
            result['preserved_prefix_maximum_difference'] != 0. or
            not replay['accepted'] or replay['substeps'] != 160 or
            result['case'] != 3 or result['tick'] != 280):
        raise ValueError('Reproduced complete fine candidate required')
    tail = read(result['configuration']['source'])
    parent = read(tail['configuration']['source_search'])
    cfg = parent['configuration']; query = read(cfg['query'])
    original = read(Path(cfg['bundle'])/'manifest.json')
    trace_path = Path(original['baseline'])/'fixed_policy_traces/case_003.npz'
    trace_manifest = read(trace_path.parent/'manifest.json')
    entry = next(r for r in trace_manifest['episodes'] if r['source_training_index'] == 3)
    if sha(trace_path) != entry['data_sha256']: raise ValueError('Changed ground-start trace')
    with np.load(trace_path, allow_pickle=False) as z:
        public = unpack_public(z['packed_public'])[280].copy()
        previous = z['dispatched_actions'][279].copy()
        student = z['dispatched_actions'][280].copy()
    limits = np.asarray(original['configuration']['interface']['limits'])
    decoded = limits[:, 0]+.5*(previous+1.)*np.diff(limits).ravel()
    if abs(public[775]-2.8)>1e-5 or np.max(abs(decoded-query['actual_previous_command']))>1e-6:
        raise ValueError('Recorded query clock or command history differs')
    rows = replay['trajectory']
    if len(rows) != 520: raise ValueError('Full original episode continuation required')
    key = 'case_003_'
    arrays = {key+'commands_rad':np.asarray([r['target'] for r in rows]),
        key+'predicted_q':np.asarray([query['initial_q']]+[r['q'] for r in rows]),
        key+'predicted_v':np.asarray([query['initial_v']]+[r['v'] for r in rows]),
        key+'predicted_contacts':np.asarray([query['initial_contacts']]+[r['contacts'] for r in rows], bool),
        key+'query_public':public, key+'previous_action':previous, key+'query_student_action':student}
    hashes[str(trace_path.resolve())] = sha(trace_path)
    output.mkdir(parents=True)
    archive = output/'sequences.npz'; np.savez_compressed(archive, **arrays)
    manifest = dict(scope=__doc__, baseline=original['baseline'], policy_sha256=original['policy_sha256'],
        configuration=original['configuration'], control_dt_s=.01,
        rows=[dict(case=3, query_tick=280, horizon_steps=520)],
        bundle_name=archive.name, bundle_sha256=sha(archive),
        ground_geometry_sha256=original['ground_geometry_sha256'],
        source_sha256=hashes, native_execution=False, valid_expert_labels=False,
        feedback_definition=dict(kind='support_tail_plus_attitude_feedback',
            fixed_early_knots_rad=result['fixed_early_knots_rad'], tail_knots_rad=result['best_knots_rad'],
            proportional=.3, damping_s=.04, maximum_joint_delta_rad=.08),
        prediction_contract='Nominal closed-loop model trajectory; native target must be recomputed from measured state.')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    return dict(bundle=str(output), archive_sha256=sha(archive), source=str(source), native_execution=False)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); print(json.dumps(export(a.source,a.output)))
