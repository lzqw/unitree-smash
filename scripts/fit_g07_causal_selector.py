#!/usr/bin/env python3
"""Fit bounded parameter selection on completed development outcomes only.

This produces a deployable checkpoint, not a physical validation result.
All options must share the same walking, planner, strike, and recovery;
only feedback gains and late arm residuals may differ.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.diagnose_g07_initial_selection import features, kernel
from quadsmash.control.causal_parameter_selector import CausalParameterSelector

ALLOWED = {'body_feedback_parameters', 'arm_pitch_output_bias_rad', 'arm_heading_gains',
           'arm_heading_reference', 'arm_joint_position_residual_rad'}


def same(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys()==right.keys() and all(same(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left)==type(right) and len(left)==len(right) and all(same(a, b) for a, b in zip(left, right))
    return left==right


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--candidate-ids', type=int, nargs='+', required=True)
    parser.add_argument('--time', type=float, default=3.)
    parser.add_argument('--ridge', type=float, required=True)
    parser.add_argument('--kernel-width', type=float, required=True)
    parser.add_argument('--calibration', type=Path, default=ROOT/'reports/G07/causal_dev_panel64_v1/calibration.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output directory')
    manifest = json.loads((args.run/'eval_manifest.json').read_text())
    if manifest.get('formal_evaluation_eligible') or not (args.run/'completed.json').exists():
        raise ValueError('Only completed development episodes may be used')
    if json.loads((args.run/'job.json').read_text())['status']!='SUCCEEDED':
        raise ValueError('Source GPU job has not completed successfully')
    calibration = json.loads(args.calibration.read_text())
    if hashlib.sha256(args.calibration.read_bytes()).hexdigest()!=manifest['calibration_sha256']:
        raise ValueError('Calibration differs from source run')
    if args.time>calibration['minimum_prepare_s']+.7 or args.ridge<=0 or args.kernel_width<=0:
        raise ValueError('Selection must precede all changed actions; positive fit parameters required')
    candidates = args.candidate_ids
    if len(set(candidates))!=len(candidates) or len(candidates)<2 or any(c<0 or c>=len(manifest['candidate_checkpoints']) for c in candidates):
        raise ValueError('Need at least two distinct recorded options')
    for c in candidates:
        source = manifest['candidate_checkpoints'][c]
        if hashlib.sha256((ROOT/source['path']).read_bytes()).hexdigest()!=source['sha256']:
            raise ValueError('Recorded option checkpoint has changed')
    checkpoints = [torch.load(ROOT/manifest['candidate_checkpoints'][c]['path'], map_location='cpu', weights_only=False) for c in candidates]
    prefix = lambda cp: {k: v for k, v in cp.items() if k not in ALLOWED}
    if any(not same(prefix(checkpoints[0]), prefix(cp)) for cp in checkpoints[1:]):
        raise ValueError('Options do not share a control prefix and fixed backbone')
    if checkpoints[0].get('replan_at_s') is not None or 'causal_parameter_selector' in checkpoints[0]:
        raise ValueError('Nested or replanning selectors require a separate validation')
    snapshots = json.loads((args.run/'preparation_snapshots.json').read_text())
    snapshot = [s for s in snapshots if abs(s['time_s']-args.time)<1e-6]
    if len(snapshot)!=1 or not all(snapshot[0]['active_first_episode']):
        raise ValueError('Original active episodes required at the declared selection time')
    public = np.zeros((len(snapshot[0]['public_frame']), 794), dtype=np.float32)
    public[:, :69] = snapshot[0]['public_frame']
    public[:, 759:776] = snapshot[0]['public_command_and_clock']
    rows = json.loads((args.run/'episodes.json').read_text())
    rows = [r for r in rows if r['candidate_id'] in candidates]
    cases = sorted({r['source_scenario'] for r in rows})
    by = {(r['candidate_id'], r['source_scenario']): r for r in rows}
    if len(by)!=len(cases)*len(candidates):
        raise ValueError('Missing or repeated option/case outcome')
    raw = [features(public[[by[c, s]['scenario'] for s in cases]], current=True) for c in candidates]
    population = np.concatenate(raw)
    mean, scale = population.mean(0), np.maximum(population.std(0), .01)
    support, alpha, intercept = [], [], []
    for c, values in zip(candidates, raw):
        x = (values-mean)/scale
        y = np.array([10*bool(by[c, s]['complete_smash'] and not by[c, s]['robot_net_contact'] and not by[c, s]['numerical_failures'])
                      +4*bool(by[c, s]['recovery_success'] and not by[c, s]['robot_net_contact'] and not by[c, s]['numerical_failures'])
                      -8*bool(by[c, s]['robot_net_contact'])-10*bool(by[c, s]['numerical_failures']) for s in cases])
        bias = y.mean()
        weights = np.linalg.solve(kernel(x, x, args.kernel_width)+args.ridge*np.eye(len(x)), y-bias)
        support.append(x); alpha.append(weights); intercept.append(bias)
    tensor = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
    tables = dict(body_feedback=torch.stack([cp.get('body_feedback_parameters', torch.zeros(3)) for cp in checkpoints]),
                  arm_pitch_bias=tensor([cp.get('arm_pitch_output_bias_rad', 0.) for cp in checkpoints]),
                  arm_heading_gains=torch.stack([cp.get('arm_heading_gains', torch.zeros(2)) for cp in checkpoints]),
                  arm_joint_residual=torch.stack([cp.get('arm_joint_position_residual_rad', torch.zeros(6)) for cp in checkpoints]))
    selector = dict(schema='g07-public-rbf-parameter-selector-v1', time_s=args.time, mean=tensor(mean), scale=tensor(scale),
                    support=tensor(support), alpha=tensor(alpha), intercept=tensor(intercept),
                    kernel_width=args.kernel_width, parameter_tables=tables)
    cp = copy.deepcopy(checkpoints[0])
    references = [option['arm_heading_reference'] for option in checkpoints if 'arm_heading_reference' in option]
    if references:
        if any(not same(references[0], reference) for reference in references[1:]):
            raise ValueError('Heading references differ')
        cp['arm_heading_reference'] = references[0]
    cp['causal_parameter_selector'] = selector
    cp['formal_acceptance'] = False
    model = CausalParameterSelector(selector, 'cpu')
    query_public = public[[by[candidates[0], s]['scenario'] for s in cases]]
    choice, scores = model.choose(torch.tensor(query_public))
    query = (features(query_public, current=True)-mean)/scale
    expected = np.stack([b+kernel(query, x, args.kernel_width)@w for x, w, b in zip(support, alpha, intercept)], 1)
    error = float(np.abs(scores.numpy()-expected).max())
    if error>1e-3:
        raise ValueError(f'Runtime versus fit score discrepancy: {error}')
    args.output.mkdir(parents=True)
    torch.save(cp, args.output/'checkpoint.pt')
    report = dict(kind='Fitted development selector only; native independent evaluation NOT RUN',
                  source_run=str(args.run), candidate_ids=candidates, common_prefix_verified=True,
                  source_checkpoints=[manifest['candidate_checkpoints'][c] for c in candidates],
                  selection_time_s=args.time, ridge=args.ridge, kernel_width=args.kernel_width,
                  runtime_public_features='log-compressed61:67,0:45,760:766; no case identifiers or future labels',
                  source_outcomes_sha256=hashlib.sha256((args.run/'episodes.json').read_bytes()).hexdigest(),
                  source_public_snapshots_sha256=hashlib.sha256((args.run/'preparation_snapshots.json').read_bytes()).hexdigest(),
                  runtime_vs_fit_max_score_difference=error,
                  checkpoint_sha256=hashlib.sha256((args.output/'checkpoint.pt').read_bytes()).hexdigest(),
                  choices_for_first_option_training_context=np.bincount(choice.numpy(), minlength=len(candidates)).tolist())
    (args.output/'fit_manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
