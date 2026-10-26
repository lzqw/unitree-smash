#!/usr/bin/env python3
"""Bind eight shared gain/body proposals to one completed native TRAIN source."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from quadsmash.control.full_racket_gain_body_adaptation_v4 import body_candidates
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases, summarize_training


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source_path, evaluation, output):
    assert not output.exists()
    inputs = json.loads((evaluation/'INPUTS.json').read_text())
    metrics = json.loads((evaluation/'metrics.json').read_text())
    job = json.loads((evaluation/'job.json').read_text())
    completed = json.loads((evaluation/'completed.json').read_text())
    assert inputs['policy_sha256'] == metrics['policy_sha256'] == sha(source_path)
    assert job['status'] == 'SUCCEEDED' and 0 < completed['env_steps'] <= job['charged_env_steps']
    assert metrics['noise_level'] == 'none' and inputs['training_indices'] == list(range(128))
    protocol_name = ('g07_full_racket_eval_v4_r3_origin0.json' if inputs.get('origin_spacing_m') == 0.
                     else 'g07_full_racket_eval_v4_r3.json')
    protocol_path = ROOT/'configs'/protocol_name
    assert sha(protocol_path) == inputs['protocol_sha256'] == metrics['protocol_sha256']
    protocol = json.loads(protocol_path.read_text()); protocol['_root'] = str(ROOT)
    cases, indices = training_cases(ROOT, protocol)
    summary = summarize_training(json.loads((evaluation/'episodes.json').read_text()), cases, indices, protocol)
    assert all(metrics[key] == value for key, value in summary.items()) and summary['numerical_failures'] == 0
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    assert source.get('native_execution') and source['asset_sha256'] == protocol['asset']['sha256']
    result = dict(created_at=datetime.now(timezone.utc).isoformat(), source_policy=str(source_path),
        source_policy_sha256=sha(source_path), source_evaluation=str(evaluation), source_summary=summary,
        source_evaluation_sha256={name:sha(evaluation/name) for name in
            ['INPUTS.json', 'metrics.json', 'episodes.json', 'job.json', 'completed.json', 'public_and_actions_first2.npz']},
        candidates=body_candidates(source), candidate_count=8, layer_order='outer then inner; each pair is leg then arm',
        hypothesis='Measured preparation is tilted before launch while residual requests are small. Compare autonomous walking, jump phase and recovery handover, retaining source and full-residual-off controls.',
        unchanged='Source weights/normalizers, physical assets, hardware, cases, criteria and all software guards.',
        all_original_training_cases_per_candidate=True, source_performance_inherited=False,
        new_native_execution=False, formal_cases_read=False, formal_acceptance=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(proposals=str(output), candidates=8, source_sha256=sha(source_path))), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--source-evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.source_evaluation, args.output)
