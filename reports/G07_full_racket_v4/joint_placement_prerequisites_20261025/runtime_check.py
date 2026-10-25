"""Restore the pinned walking actor and check the real candidate CPU loader."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from quadsmash.control.full_racket_residual_v4 import make_rally_controller
from quadsmash.control import go2_walk_transfer, piper_geometry
from quadsmash.usd_dependencies_v4 import verify_dependency_manifest


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    output = HERE / 'RUNTIME_CHECK.json'
    assert source != ROOT and not output.exists()
    restore_path = HERE / 'RESTORATION.json'
    restore = json.loads(restore_path.read_bytes())
    assert restore['restored_historical_prerequisite_verification']['status'] == 'PASSED'
    for item in restore['restored']:
        assert sha(ROOT / item['path']) == item['sha256']
    walk_manifest = ROOT / 'configs/g07_go2_walk_transfer.json'
    walking = json.loads(walk_manifest.read_bytes())
    target = ROOT / walking['policy_path']
    original = source / walking['policy_path']
    assert sha(original) == walking['policy_sha256']
    copied = False
    if not target.exists():
        assert not target.is_symlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + f'.restore-{os.getpid()}.tmp')
        try:
            with temporary.open('xb') as stream:
                stream.write(original.read_bytes())
            assert sha(temporary) == walking['policy_sha256']
            os.link(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        copied = True
    assert sha(target) == walking['policy_sha256']
    prior_check = source / 'reports/G07_full_racket_v4/formal_residual_loader_check_20261025/REPORT.json'
    prior = json.loads(prior_check.read_bytes())
    policy_path = source / prior['loaded_policy_path']
    checkpoint_path = source / prior['canonical_checkpoint_path']
    assert sha(policy_path) == prior['policy_sha256']
    assert sha(checkpoint_path) == prior['canonical_checkpoint_sha256']
    assert json.loads((policy_path.parent / 'job.json').read_bytes())['status'] == 'SUCCEEDED'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    policy = torch.load(policy_path, map_location='cpu', weights_only=False)
    assert checkpoint['iter'] == 32 and checkpoint['native']['canonical_restart']
    base_path = ROOT / 'reports/G07_full_racket_v4/cem32_lateral_feedback_1015/best_candidate.pt'
    assert sha(base_path) == policy['residual_training_metadata']['base_policy_sha256']
    sample = checkpoint['native']['environment']['tensors']['final_observation'][:200].clone()
    assert sample.shape == (200, 794) and torch.isfinite(sample).all()
    assert piper_geometry.ROOT == ROOT
    assert Path(go2_walk_transfer.__file__).resolve().is_relative_to(ROOT)
    dependencies = verify_dependency_manifest(ROOT)
    controller = make_rally_controller(policy, 200, 'cpu')
    assert type(controller).__name__ == 'FullRacketResidualControllerV4'
    assert controller.base.parameters.shape[0] == len(controller.mixer.nonzero_seen) == 200
    with torch.inference_mode():
        actual = controller.actor(sample)
        split = torch.cat([controller.actor(sample[:128]), controller.actor(sample[128:])])
    assert actual.shape == (200, 18) and torch.isfinite(actual).all()
    error = float((actual - split).abs().max())
    assert torch.allclose(actual, split, atol=2e-5, rtol=2e-5)
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(),
        scope='Actual candidate-tree CPU controller construction and public actor inference on recorded canonical TRAIN observations. No Isaac/PhysX execution or new policy performance.',
        historical_restoration_sha256=sha(restore_path),
        restored_walking_actor=dict(path=walking['policy_path'], source=str(original),
            bytes=target.stat().st_size, sha256=sha(target), copied_on_this_run=copied,
            manifest_sha256=sha(walk_manifest)),
        source_check_report_sha256=sha(prior_check), source_iteration=32,
        policy_sha256=sha(policy_path), canonical_checkpoint_sha256=sha(checkpoint_path),
        candidate_base_sha256=sha(base_path), candidate_module_root=str(ROOT),
        controller_factory='quadsmash.control.full_racket_residual_v4.make_rally_controller',
        actual_controller_type=type(controller).__name__, controller_lanes=200,
        actual_observation_source='Actual canonical32 native.environment.tensors.final_observation[:200]',
        public_observation_shape=list(sample.shape), actor_output_shape=list(actual.shape),
        all_probe_inputs_outputs_finite=True, split_batch_max_error=error,
        asset_dependency_layers_verified=len(dependencies['layers']),
        native_composition_executed=False, candidate_training_executed=False,
        canonical96_transfer_checked=False, formal_cases_read=False,
        formal_acceptance=False, goal_complete=False, script_sha256=sha(Path(__file__).resolve()),
        remaining='Review completed original96/fixed assessment and deliver raw data/videos, reconcile all source owners and use the single GPU0 guard, then validate actual native composition and ground initialization before candidate learning. This CPU check starts no GPU workflow.')
    temporary = output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(output)
    print(json.dumps(dict(walking_actor_restored=copied, candidate_controller_lanes=200,
        actor_output_shape=list(actual.shape), split_batch_max_error=error,
        asset_dependency_layers_verified=len(dependencies['layers']),
        candidate_training_executed=False, receipt=str(output))))


if __name__ == '__main__':
    main()
