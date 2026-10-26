"""Check extracted intermediate CPU evidence using its own bundled sources."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    assert not args.output.exists()
    manifest = read(root/'CPU_EXPORT_MANIFEST.json')
    for name, item in manifest['payloads'].items():
        path = root/name
        assert path.is_file() and path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], name
    sys.path[:0] = [str(root), str(root/'vendor/rsl_rl')]
    import torch
    import rsl_rl
    assert Path(rsl_rl.__file__).resolve().is_relative_to(root/'vendor/rsl_rl')
    q = root/'reports/G07_full_racket_v4/centered_placement075_intermediate_selection_queue01'
    spec = importlib.util.spec_from_file_location('restored075_export', q/'export_diagnostic.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.ROOT == root
    reference = module.reference_check()
    sample, _ = module.actual_samples()
    source = torch.load(module.SOURCE/'checkpoint.pt', map_location='cpu', weights_only=False)
    checked = []
    from tensordict import TensorDict
    for iteration in [16, 24]:
        path, receipt = module.export(iteration)  # Existing export: hash checks, no rewriting.
        native = torch.load(module.RUN/'ppo'/f'model_{iteration-1}.pt', map_location='cpu', weights_only=False)
        assert native['iter'] == iteration and not native['native']['canonical_restart']
        assert native['native']['total_timesteps'] == iteration*2048*128
        assert int(native['model_state_dict']['actor_obs_normalizer.count']) == 10485760+iteration*2048*128
        assert int(native['model_state_dict']['critic_obs_normalizer.count']) == iteration*2048*128
        before, after = source['optimizer_state_dict']['state'], native['optimizer_state_dict']['state']
        assert len(after) == 17 and set(before) == set(after)
        assert all(float(after[k]['step'])-float(before[k]['step']) == (iteration-8)*20 for k in before)
        public = torch.load(path, map_location='cpu', weights_only=False)
        actor = torch.jit.load(io.BytesIO(public['residual_actor_jit']), map_location='cpu').eval()
        policy = module.policy_from_native(native)
        with torch.inference_mode():
            expected = policy.act_inference(TensorDict({'policy': sample}, batch_size=[len(sample)]))
            error = float((actor(sample)-expected).abs().max())
        assert error == 0. and not public['formal_acceptance'] and not public['performance_inherited']
        checked.append(dict(iteration=iteration, actual_public_samples=len(sample),
            independently_rebuilt_actor_max_error=error, policy_sha256=sha(path),
            actor_RMS_count=receipt['actor_RMS_count'], critic_RMS_count=receipt['critic_RMS_count'],
            diagnostic_is_canonical_restart=False))
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), extraction_root=str(root),
        all_payload_SHA_and_sizes_checked=len(manifest['payloads']), bundled_RSL_source_used=str(Path(rsl_rl.__file__).resolve()),
        source8_reference=reference, intermediate_actor_checks=checked,
        scope='Fresh extracted CPU data/source/weight/export fidelity only. No Isaac/PhysX execution or frozen-policy performance.',
        new_native_execution=False, formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(payloads_checked=len(manifest['payloads']),
        actor_errors=[e['independently_rebuilt_actor_max_error'] for e in checked], bundled_RSL_source_used=True,
        new_native_execution=False)))


if __name__ == '__main__':
    main()
