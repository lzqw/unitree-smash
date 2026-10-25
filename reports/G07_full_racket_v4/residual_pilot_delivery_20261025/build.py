"""Verify and archive the completed native residual PPO pilot without inheriting scores."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REPORTS = ROOT / 'reports/G07_full_racket_v4'
RUN = REPORTS / 'residual_ppo_pilot_train128_1230'
BASE = REPORTS / 'cem32_lateral_feedback_1015/best_candidate.pt'
ARCHIVE = REPORTS / 'G07_native_residual_PPO_iter8_checkpoint_and_records_20261025.zip'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main():
    assert not ARCHIVE.exists(), 'Never overwrite a delivered native archive'
    import torch
    import numpy as np
    from tensordict import TensorDict
    from rsl_rl.modules import ActorCritic
    import sys
    sys.path.insert(0, str(ROOT))
    from quadsmash.evaluation_v4 import independent_episode
    from quadsmash.usd_dependencies_v4 import verify_dependency_manifest
    torch.set_num_threads(2)
    module_spec = importlib.util.spec_from_file_location('resume_checks', REPORTS / 'residual_resume24_queue01/run.py')
    checks = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(checks)
    verified = checks.verify_training(RUN, 8, 0)
    protocol = read(ROOT / 'configs/g07_full_racket_eval_v4_r3.json')
    protocol['_root'] = str(ROOT)
    rows = read(RUN / 'episodes.json')
    outcomes = [independent_episode(row, protocol) for row in rows]
    counts = dict(completed_episodes=len(rows), air=sum(r['airborne_contact'] for r in outcomes),
        conditional_recovery_count=sum(r['conditional_recovery'] for r in outcomes),
        complete=sum(r['complete'] for r in outcomes), numerical_failures=sum(r['numerical_failures'] for r in rows))
    assert counts == read(RUN / 'metrics.json')['counts']
    inputs = read(RUN / 'INPUTS.json')
    assert inputs['training_indices'] == list(range(128)) and not inputs['formal_cases_read']
    source_hashes = read(RUN / 'source_hashes.json')
    critical = sorted(ROOT.glob('quadsmash/**/*.py')) + sorted(ROOT.glob('configs/*.json')) + [ROOT / 'scripts/train_full_racket_residual_ppo_v4.py']
    assert all(sha(p) == source_hashes[str(p.relative_to(ROOT))] for p in critical)
    dependencies = verify_dependency_manifest(ROOT)
    native = torch.load(RUN / 'checkpoint.pt', map_location='cpu', weights_only=False)
    exported = torch.load(RUN / 'policy.pt', map_location='cpu', weights_only=False)
    public_path = REPORTS / 'train128_lateral_c12_1025/public_and_actions_first2.npz'
    with np.load(public_path) as data:
        ids = np.flatnonzero(data['valid_first_episode'][:400].reshape(-1))[::16]
        sample = torch.from_numpy(data['observations'][:400].reshape(-1, 794)[ids].copy())
    obs = TensorDict({'policy': sample}, batch_size=[len(sample)])
    configuration = native['native']['training_config']
    policy = ActorCritic(obs, configuration['obs_groups'], 18,
        **{k: v for k, v in configuration['policy'].items() if k != 'class_name'}).eval()
    policy.load_state_dict(native['model_state_dict'])
    portable = torch.jit.load(io.BytesIO(exported['residual_actor_jit']), map_location='cpu').eval()
    with torch.inference_mode():
        expected = policy.act_inference(obs)
        actual = portable(sample)
        error = float((actual - expected).abs().max())
        assert torch.allclose(expected, actual, atol=2e-5, rtol=2e-5)
    fidelity = dict(scope='Read-only CPU actor equivalence on previously recorded baseline public states; no new native performance.',
        samples=len(sample), public_source=str(public_path.relative_to(ROOT)), public_source_sha256=sha(public_path),
        selected_flat_rows=ids.tolist(), max_absolute_error=error, comparison_device='cpu',
        native_export_comparison=exported['residual_training_metadata'],
        max_raw_residual_on_sample=float(actual.abs().max()), formal_acceptance=False)
    write(HERE / 'ACTOR_CPU_FIDELITY.json', fidelity)
    report = dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='Completed actual native residual PPO pilot. Changing-policy TRAIN counts only; frozen-policy performance remains unproven.',
        source_run=str(RUN.relative_to(ROOT)), job=read(RUN / 'job.json'),
        verification=verified, independent_closed_training_counts=counts,
        distinct_TRAIN_indices=len(set(row['source_training_index'] for row in rows)),
        actual_control_env_steps=131072, physics_dt_s=inputs['physics_dt_s'], policy_dt_s=inputs['policy_dt_s'],
        warmup_updates=2, learned_actor_updates=6, all_update_losses_and_model_optimizer_tensors_finite=True,
        canonical_ground_checkpoint=True, checkpoint_aborted_episodes=read(RUN / 'metrics.json')['checkpoint_aborted_episodes'],
        all10_USD_dependencies_verified=True, actor_export_error_on_original_GPU=exported['residual_training_metadata']['export_actor_max_error'],
        frozen_TRAIN128_output='reports/G07_full_racket_v4/residual_ppo_fixed_train128_1250',
        frozen_policy_evaluation_status_at_report='RUNNING; inspect its actual guard and receipt before making claims',
        formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    write(HERE / 'REPORT.json', report)
    paths = [p for p in RUN.iterdir() if p.is_file() and 'partial' not in p.name]
    paths += [BASE, HERE / 'REPORT.json', HERE / 'ACTOR_CPU_FIDELITY.json', Path(__file__)]
    entries = {str(p.relative_to(ROOT)): p for p in paths}
    layers = read(RUN / 'asset_dependencies.json')['resolved_layers']
    walk = read(ROOT / 'configs/g07_go2_walk_transfer.json')
    walk_path = (ROOT / walk['policy_path']).resolve()
    assert sha(walk_path) == walk['policy_sha256']
    for p in [*[Path(v) for v in layers], walk_path]:
        name = str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else 'external_dependencies/' + str(p.relative_to('/'))
        entries[name] = p
    manifest = dict(scope=report['scope'], formal_acceptance=False, learner_formal_cases_read=False,
        policy_sha256=verified['policy_sha256'], checkpoint_sha256=verified['checkpoint_sha256'],
        actual_GPU_hours=report['job']['charged_gpu_hours'], actual_control_env_steps=131072,
        runtime_note='Use the original source_snapshot.zip from this native run. checkpoint.pt is the canonical optimizer/normalizer/RNG ground restart; policy.pt is portable actor plus embedded baseline. Diagnostic model_i checkpoints are intentionally excluded.',
        dependency_note='Restore external_dependencies paths to the original recorded layout or explicitly remap them. The separately pinned Isaac runtime is required and is not bundled.',
        all_USD_layers=dependencies['layers'], baseline_walk_actor_sha256=walk['policy_sha256'],
        files={name: dict(bytes=p.stat().st_size, sha256=sha(p)) for name, p in sorted(entries.items())})
    write(HERE / 'MANIFEST.json', manifest)
    with zipfile.ZipFile(ARCHIVE, 'w', zipfile.ZIP_DEFLATED, compresslevel=5) as archive:
        archive.writestr('DELIVERY_MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
        for name, p in sorted(entries.items()):
            archive.write(p, name)
    with zipfile.ZipFile(ARCHIVE) as archive:
        assert archive.testzip() is None
        for name, value in manifest['files'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == value['sha256']
    result = dict(archive=str(ARCHIVE.relative_to(ROOT)), sha256=sha(ARCHIVE), bytes=ARCHIVE.stat().st_size,
        members=len(entries) + 1, all_archive_members_rehashed=True, formal_acceptance=False)
    write(HERE / 'ARCHIVE.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
