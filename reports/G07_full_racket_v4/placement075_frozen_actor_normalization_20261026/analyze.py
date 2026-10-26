"""Separate recorded actor-weight and normalization changes on fixed real inputs.

Hybrids are CPU diagnostics in memory only. No policy is exported or executed
in native physics, and this does not establish a causal task-outcome effect.
"""
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REPORTS = ROOT / 'reports/G07_full_racket_v4'
sys.path.insert(0, str(ROOT))
from quadsmash.rl.episode_trace_v4 import unpack_public
from scripts.initialize_centered_policy_v4 import same_tree


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    output = HERE / 'ACTUAL_FIXED_INPUT_DECOMPOSITION.json'
    assert not output.exists()
    torch.set_num_threads(2)
    paths = {8: REPORTS/'residual_ppo_centered_placement075_n2048_probe8_01/policy.pt',
        16: REPORTS/'centered_placement075_intermediate_selection_queue01/exports/iter16/policy.pt',
        40: REPORTS/'residual_ppo_centered_placement075_n2048_resume32_02/policy.pt'}
    held = {ROOT/name: digest for name, digest in read(REPORTS/'centered_placement075_intermediate_selection_queue01/PREDECLARED_PLAN.json')['source_hashes'].items()}
    assert all(sha(path) == digest for path, digest in held.items())
    source_files = [Path(__file__), *paths.values()]
    trace = REPORTS / 'residual_ppo_centered_placement075_n2048_resume32_02/trajectory_traces'
    manifest = read(trace/'manifest.json');source_files.append(trace/'manifest.json')
    batches = []
    for item in manifest['saved_episodes']:
        path = trace/item['data_file'];meta = read(path.with_suffix('.json'))
        assert sha(path) == item['data_sha256'] == meta['data_sha256']
        with np.load(path) as data:public = unpack_public(data['packed_public'])
        assert hashlib.sha256(public.tobytes()).hexdigest() == meta['full_public794_sha256']
        batches.append(torch.from_numpy(public));source_files += [path, path.with_suffix('.json')]
    inputs = torch.cat(batches);assert inputs.shape == (7436, 794)
    checkpoints = {iteration: torch.load(path, map_location='cpu', weights_only=False) for iteration, path in paths.items()}
    actors = {iteration: torch.jit.load(io.BytesIO(checkpoint['residual_actor_jit']), map_location='cpu').eval()
              for iteration, checkpoint in checkpoints.items()}
    assert all(same_tree(checkpoint['base_controller'], checkpoints[8]['base_controller']) for checkpoint in checkpoints.values())
    assert all(checkpoint['residual_joint_scale_rad'] == [.03]*12+[.15]*6 for checkpoint in checkpoints.values())
    scale = torch.tensor([.03]*12+[.15]*6)
    masks = dict(all=torch.ones(len(inputs), dtype=torch.bool),
                 prepare=inputs[:,775] < 2., intercept=(inputs[:,775]>=2.) & (inputs[:,775]<3.2),
                 after_intercept=inputs[:,775]>=3.2)

    def metrics(delta):
        result = {}
        for name, mask in masks.items():
            group = delta[mask] * scale
            result[name] = dict(samples=int(mask.sum()), leg_RMS_requested_rad=float(group[:,:12].square().mean().sqrt()),
                arm_RMS_requested_rad=float(group[:,12:].square().mean().sqrt()),
                max_abs_requested_rad=float(group.abs().max()))
        return result

    with torch.no_grad():
        original = actors[8](inputs)
        results = []
        for iteration in [16, 40]:
            state = actors[iteration].state_dict();reference = actors[8].state_dict();assert state.keys()==reference.keys()
            weight_only = torch.jit.load(io.BytesIO(checkpoints[8]['residual_actor_jit']), map_location='cpu').eval()
            norm_only = torch.jit.load(io.BytesIO(checkpoints[8]['residual_actor_jit']), map_location='cpu').eval()
            weight_only.load_state_dict({name: reference[name] if name.startswith('0.') else state[name] for name in state})
            norm_only.load_state_dict({name: state[name] if name.startswith('0.') else reference[name] for name in state})
            current, weights, normalization = actors[iteration](inputs), weight_only(inputs), norm_only(inputs)
            assert all(torch.isfinite(value).all() for value in [original, current, weights, normalization])
            a,b,c,d = [value.clamp(-1,1) for value in [original,current,weights,normalization]]
            results.append(dict(iteration=iteration, actor_RMS_count=int(state['0.count']),
                original_RMS_count=int(reference['0.count']), actual_actor_delta=metrics(b-a),
                weights_changed_normalizer8=metrics(c-a), normalizer_changed_weights8=metrics(d-a),
                normalizer_effect_given_current_weights=metrics(b-c), interaction=metrics(b-c-d+a),
                unclipped_mean_outside_action_range_fraction=float((current.abs()>1).float().mean())))
    assert all(sha(path) == digest for path, digest in held.items())
    record = dict(checked_at=datetime.now(timezone.utc).isoformat(), actual_saved_episodes=17,
        actual_fixed_public794_inputs=len(inputs), comparisons=results,
        sources={str(path.relative_to(ROOT)):sha(path) for path in source_files},
        physical_units='Requested residual joint offsets before joint limits and arm rate limiting',
        scope='CPU actor function decomposition on saved native inputs. No hybrid-native execution or causal task-outcome claim.',
        hybrid_policies_exported=False, main_held_sources_unchanged=True,
        new_native_execution=False, formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    output.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(dict(samples=len(inputs),comparisons=results)), flush=True)


if __name__ == '__main__':main()
