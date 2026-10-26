"""Freeze actual placement075 intermediate actors for inference, never resume."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.modules import ActorCritic

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REPORTS = ROOT/'reports/G07_full_racket_v4'
RUN = REPORTS/'residual_ppo_centered_placement075_n2048_resume32_02'
SOURCE = REPORTS/'residual_ppo_centered_placement075_n2048_probe8_01'
BASE = REPORTS/'residual_ppo_joint_placement_actor32_n2048_probe8_01/policy.pt'
CONFIG = ROOT/'configs/g07_full_racket_residual_ppo_centered_placement075_v4.json'
SAMPLE = REPORTS/'video_snapshot_placement075_first_complete_20261026'
STEPS_PER_UPDATE = 2048*128
PRIOR_ACTOR_RMS_OFFSET = 10485760
sys.path.insert(0, str(ROOT))
from quadsmash.rl.episode_trace_v4 import unpack_public
from quadsmash.rl.full_racket_residual_ppo_v4 import export_residual_policy


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def actual_samples():
    manifest = read(SAMPLE/'cases.json')
    assert [(c['env_id'], c['episode_index']) for c in manifest['cases']] == [(208, 4), (23, 2), (8, 3)]
    batches, sources = [], [SAMPLE/'cases.json']
    for case in manifest['cases']:
        path = SAMPLE/case['run']/f'lane{case["env_id"]:04d}_episode{case["episode_index"]:06d}.npz'
        meta = read(path.with_suffix('.json'))
        assert meta['native_execution'] and meta['data_sha256'] == sha(path)
        with np.load(path) as data:
            public = unpack_public(data['packed_public'])
        assert public.shape == (meta['samples'], 794)
        assert hashlib.sha256(public.tobytes()).hexdigest() == meta['full_public794_sha256']
        batches.append(torch.from_numpy(public))
        sources.extend([path, path.with_suffix('.json')])
    sample = torch.cat(batches)
    assert len(sample) >= 800 and torch.isfinite(sample).all()
    return sample, sources


def policy_from_native(state):
    config = read(CONFIG)
    options = dict(config['runner']['policy'])
    options.pop('class_name')
    obs = TensorDict({'policy': torch.zeros(2, 794)}, batch_size=[2])
    with contextlib.redirect_stdout(io.StringIO()):
        policy = ActorCritic(obs, config['runner']['obs_groups'], 18, **options)
    policy.load_state_dict(state['model_state_dict'], strict=True)
    policy.eval()
    assert all(torch.isfinite(v).all() for v in policy.state_dict().values() if torch.is_tensor(v))
    return policy


def reference_check():
    sample, sources = actual_samples()
    source = torch.load(SOURCE/'checkpoint.pt', map_location='cpu', weights_only=False)
    assert source['iter'] == 8 and source['native']['canonical_restart']
    assert sha(SOURCE/'checkpoint.pt') == read(RUN/'INPUTS.json')['resume_checkpoint_sha256']
    model = source['model_state_dict']
    assert int(model['actor_obs_normalizer.count']) == PRIOR_ACTOR_RMS_OFFSET+8*STEPS_PER_UPDATE
    assert int(model['critic_obs_normalizer.count']) == 8*STEPS_PER_UPDATE
    original = torch.load(SOURCE/'policy.pt', map_location='cpu', weights_only=False)
    policy = policy_from_native(source)
    actor = torch.jit.load(io.BytesIO(original['residual_actor_jit']), map_location='cpu').eval()
    with torch.inference_mode():
        expected = policy.act_inference(TensorDict({'policy': sample}, batch_size=[len(sample)]))
        error = float((expected-actor(sample)).abs().max())
    assert error == 0., 'Rebuilt075 canonical8 mean/normalizer differs from its completed public export'
    return dict(actual_public_samples=len(sample), actual_source8_actor_export_max_error=error,
        source_canonical8_sha256=sha(SOURCE/'checkpoint.pt'), source_public8_sha256=sha(SOURCE/'policy.pt'),
        source_actor_RMS_count=int(model['actor_obs_normalizer.count']),
        source_critic_RMS_count=int(model['critic_obs_normalizer.count']),
        prior_actor_RMS_offset=PRIOR_ACTOR_RMS_OFFSET,
        native_restart_created=False, new_native_execution=False,
        sample_source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sources})


def export(iteration):
    assert iteration in [16, 24, 32]
    directory = HERE/'exports'/f'iter{iteration:02d}'
    output, receipt = directory/'policy.pt', directory/'EXPORT_CHECK.json'
    diagnostic = RUN/'ppo'/f'model_{iteration-1}.pt'
    if output.exists():
        checked = read(receipt)
        assert checked['policy_sha256'] == sha(output)
        assert checked['source_diagnostic_sha256'] == sha(diagnostic)
        assert all(sha(ROOT/name) == digest for name, digest in checked['source_sha256'].items())
        return output, checked
    assert not receipt.exists()
    progress = read(RUN/'progress.json')
    assert progress['update'] > iteration and progress['numerical_failures'] == 0
    inputs = read(RUN/'INPUTS.json')
    assert inputs['configuration_sha256'] == sha(CONFIG)
    assert inputs['contract']['base_policy_sha256'] == sha(BASE)
    assert inputs['contract']['base_controller_mode'] == 'frozen_public_policy'
    assert inputs['contract']['joint_scale_rad'] == [.03]*12+[.15]*6
    source = torch.load(SOURCE/'checkpoint.pt', map_location='cpu', weights_only=False)
    assert inputs['resume_checkpoint_sha256'] == sha(SOURCE/'checkpoint.pt')
    assert inputs['initialization'] == dict(mode='strict_canonical_ground_resume', num_envs=2048, initial_iteration=8)
    assert source['infos']['residual_contract'] == inputs['contract']
    state = torch.load(diagnostic, map_location='cpu', weights_only=False)
    assert state['iter'] == iteration and not state['native']['canonical_restart']
    assert state['native']['training_signature'] == source['native']['training_signature']
    assert state['native']['environment']['num_envs'] == 2048
    assert state['native']['total_timesteps'] == iteration*STEPS_PER_UPDATE
    # RSL intermediate runner saves have infos=None; provenance comes from the
    # real job INPUTS, strict source SHA/signature, counters and Adam continuity.
    assert state['infos'] is None
    before, after = source['optimizer_state_dict']['state'], state['optimizer_state_dict']['state']
    assert set(before) == set(after) and len(after) == 17
    assert all(float(after[k]['step'])-float(before[k]['step']) == (iteration-8)*20 for k in before)
    assert all(torch.isfinite(v).all() for part in after.values() for v in part.values() if torch.is_tensor(v))
    model = state['model_state_dict']
    assert int(model['actor_obs_normalizer.count']) == PRIOR_ACTOR_RMS_OFFSET+iteration*STEPS_PER_UPDATE
    assert int(model['critic_obs_normalizer.count']) == iteration*STEPS_PER_UPDATE
    for name in ['actor_obs_normalizer.count', 'critic_obs_normalizer.count']:
        assert int(model[name])-int(source['model_state_dict'][name]) == (iteration-8)*STEPS_PER_UPDATE
    sample, sources = actual_samples()
    policy = policy_from_native(state)
    base = torch.load(BASE, map_location='cpu', weights_only=False)
    result = export_residual_policy(base, policy, read(CONFIG)['joint_scale_rad'], sample,
        dict(native_execution=True, completed_updates=iteration, base_policy_sha256=sha(BASE),
            actual_control_env_steps=(iteration-8)*STEPS_PER_UPDATE,
            cumulative_placement075_control_env_steps=iteration*STEPS_PER_UPDATE,
            base_controller_mode='frozen_public_policy', num_envs=2048,
            initialization=inputs['initialization'], source_diagnostic_sha256=sha(diagnostic),
            source_canonical_restart=False, CPU_export_of_actual_executed_native_actor=True,
            frozen_mean_policy_performance_inherited=False))
    assert result['base_controller']['residual_actor_jit'] == base['residual_actor_jit']
    assert not result['performance_inherited'] and not result['formal_acceptance']
    actor = torch.jit.load(io.BytesIO(result['residual_actor_jit']), map_location='cpu').eval()
    with torch.inference_mode():
        expected = policy.act_inference(TensorDict({'policy': sample}, batch_size=[len(sample)]))
        error = float((actor(sample)-expected).abs().max())
    assert error <= 2e-5
    directory.mkdir(parents=True, exist_ok=False)
    temporary = output.with_suffix('.pt.tmp')
    torch.save(result, temporary)
    temporary.replace(output)
    checked = dict(iteration=iteration, source_diagnostic=str(diagnostic.relative_to(ROOT)),
        source_diagnostic_sha256=sha(diagnostic), policy_sha256=sha(output),
        configuration_sha256=sha(CONFIG), fixed_base_policy_sha256=sha(BASE),
        actual_public_samples=len(sample), actor_export_max_error=error,
        actor_RMS_count=int(model['actor_obs_normalizer.count']),
        critic_RMS_count=int(model['critic_obs_normalizer.count']), prior_actor_RMS_offset=PRIOR_ACTOR_RMS_OFFSET,
        all17_Adam_states_preserved_and_advanced=(iteration-8)*20,
        original_normalization_preserved=True, one_fixed_base_and_one_outer_residual=True,
        native_source_training_and_subsequent_collection_verified=True,
        CPU_export_only=True, new_native_execution=False, source_is_canonical_restart=False,
        scope='Actual placement075 native intermediate actor frozen for inference. Diagnostic weights are never restart checkpoints or inherited performance.',
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in
            [Path(__file__), diagnostic, CONFIG, BASE, SOURCE/'checkpoint.pt', SOURCE/'policy.pt', RUN/'INPUTS.json', *sources]},
        formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    receipt.write_text(json.dumps(checked, indent=2)+'\n')
    return output, checked
