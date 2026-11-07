"""Export independently replayed collision-constrained support candidates.

This prepares a native experiment, not a native success or an expert label.
The old refined exporter supplies only the fixed controller/reference template;
its weaker collision checks are not evidence for accepting the new candidate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_native_refined_tracking_v1 import prepare_refined


def validated_candidate(search):
    if (search.get('native_execution') is not False or
            search.get('valid_expert_labels') is not False or
            search.get('full_model_candidate_found') is not True or
            search.get('baseline_reproduced') is not True):
        raise ValueError('Completed model candidate with independent replay required')
    best, replay = search['best'], search['independent_replay']
    if not isinstance(replay, dict):
        raise ValueError('Missing independent replay')
    trajectories = []
    for row in [best, replay]:
        r = row['result']
        if (row['substeps'] != 160 or r['substeps'] != 160 or
                r['accepted'] is not True or r['failure'] is not None or
                r['completed_model_steps'] != 520 or r['required_model_steps'] != 520 or
                r['terminal_violations'] or r['trailing_model_window_s'] < .8 or
                r['native_execution'] is not False or r['valid_expert_labels'] is not False):
            raise ValueError('Full fine-model episode and stability required')
        trajectory = r['trajectory']
        if len(trajectory) != 520:
            raise ValueError('Incomplete trajectory')
        values = {}
        for field, width in [('q', 25), ('v', 24), ('target', 18), ('contacts', 4)]:
            values[field] = np.asarray([step[field] for step in trajectory])
            if values[field].shape != (520, width) or not np.isfinite(values[field]).all():
                raise ValueError('Malformed trajectory ' + field)
        trajectories.append(values)
    knots = np.asarray(best['knots_rad'])
    if knots.shape != (6, 18) or not np.isfinite(knots).all():
        raise ValueError('Six finite full-body knots required')
    if not np.array_equal(knots, replay['knots_rad']):
        raise ValueError('Replay uses different actions')
    if not np.array_equal(knots[:3], search['fixed_early_knots_rad']):
        raise ValueError('Declared fixed prefix changed')
    for field in trajectories[0]:
        if not np.array_equal(trajectories[0][field], trajectories[1][field]):
            raise ValueError('Independent trajectory differs: ' + field)
    return knots, trajectories[1]


def prepare_constrained(root, output, bundle_output, base_bundle, search_path):
    root, output, bundle_output, base_bundle, search_path = [Path(p).resolve() for p in
        (root, output, bundle_output, base_bundle, search_path)]
    read = lambda p: json.loads(Path(p).read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    search = read(search_path)
    knots, trajectory = validated_candidate(search)
    for p, digest in search['source_sha256'].items():
        if sha(p) != digest:
            raise ValueError('Changed candidate source ' + p)
    previous_path = Path(search['configuration']['source'])
    previous = read(previous_path)
    if previous['margin_rad'] != .02 or previous['substeps'] != 160:
        raise ValueError('Unexpected constrained prediction settings')
    old_search_path = Path(previous['configuration']['search'])
    result = prepare_refined(root, output, bundle_output, base_bundle, old_search_path)
    manifest_path = bundle_output / 'manifest.json'
    manifest = read(manifest_path)
    archive = bundle_output / manifest['bundle_name']
    with np.load(archive, allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in data.files}
    key = 'case_003'
    # Fixed q/v tracking references remain untouched. These are predictions of
    # the new controller, not references to feed back into that controller.
    arrays[key + '_commands_rad'] = trajectory['target']
    for suffix in ['q', 'v', 'contacts']:
        name = key + '_predicted_' + suffix
        arrays[name] = np.vstack([arrays[name][0], trajectory[suffix]])
    with archive.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    manifest['bundle_sha256'] = sha(archive)
    manifest['feedback_definition'].update(
        fixed_early_knots_rad=knots[:3].tolist(), tail_knots_rad=knots[3:].tolist())
    manifest['scope'] = 'Constrained full-episode model candidate; native execution and robustness unverified.'
    definition = manifest['reference_tracking_definition']
    definition.update(model_result=str(search_path), model_result_sha256=sha(search_path),
        model_probes_passed=1, model_probes_total=1, model_independent_replay=True,
        full_self_collision_certification=False, hip_mount_margin_rad=.02)
    manifest['source_sha256'].update(search['source_sha256'])
    added = {str(p): sha(p) for p in [search_path, Path(__file__).resolve()]}
    manifest['source_sha256'].update(added)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    result.update(bundle_manifest_sha256=sha(manifest_path), model_search=str(search_path),
        model_probes_passed=1, model_probes_total=1, model_independent_replay=True,
        teacher_acceptance=False, full_self_collision_certification=False)
    result['source_sha256'].update(search['source_sha256'])
    result['source_sha256'].update(added)
    (output / 'preparation.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    for key in ['output', 'bundle-output', 'base-bundle', 'search']:
        parser.add_argument('--' + key, type=Path, required=True)
    a = parser.parse_args()
    print(json.dumps(prepare_constrained(a.root, a.output, a.bundle_output, a.base_bundle, a.search)))
