"""Export actual native intermediate actors for inference, never for resume."""
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
RUN = REPORTS/'residual_ppo_centered_base8_n2048_resume32_01'
SOURCE = REPORTS/'residual_ppo_centered_base8_n2048_probe8_02'
BASE = REPORTS/'residual_ppo_joint_placement_actor32_n2048_probe8_01/policy.pt'
CONFIG = ROOT/'configs/g07_full_racket_residual_ppo_centered_v4.json'
SAMPLE = REPORTS/'video_snapshot_centered_resume_first_complete_20261026'
sys.path.insert(0,str(ROOT))
from quadsmash.rl.episode_trace_v4 import unpack_public
from quadsmash.rl.full_racket_residual_ppo_v4 import export_residual_policy


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def actual_samples():
    manifest = read(SAMPLE/'cases.json'); batches = []; sources = [SAMPLE/'cases.json']
    for case in manifest['cases']:
        path = SAMPLE/case['run']/f'lane{case["env_id"]:04d}_episode{case["episode_index"]:06d}.npz'
        meta = read(path.with_suffix('.json'))
        assert meta['native_execution'] and meta['data_sha256'] == sha(path)
        with np.load(path) as data:public = unpack_public(data['packed_public'])
        assert hashlib.sha256(public.tobytes()).hexdigest() == meta['full_public794_sha256']
        batches.append(torch.from_numpy(public)); sources.extend([path,path.with_suffix('.json')])
    sample = torch.cat(batches)
    assert sample.shape == (1890,794) and torch.isfinite(sample).all()
    return sample,sources


def policy_from_native(state):
    config = read(CONFIG); options = dict(config['runner']['policy']); options.pop('class_name')
    obs = TensorDict({'policy':torch.zeros(2,794)},batch_size=[2])
    with contextlib.redirect_stdout(io.StringIO()):
        policy = ActorCritic(obs,config['runner']['obs_groups'],18,**options)
    policy.load_state_dict(state['model_state_dict'],strict=True); policy.eval()
    assert all(torch.isfinite(value).all() for value in policy.state_dict().values() if torch.is_tensor(value))
    return policy


def reference_check():
    sample,sources = actual_samples()
    source = torch.load(SOURCE/'checkpoint.pt',map_location='cpu',weights_only=False)
    assert source['iter'] == 8 and source['native']['canonical_restart']
    assert sha(SOURCE/'checkpoint.pt') == read(RUN/'INPUTS.json')['resume_checkpoint_sha256']
    original = torch.load(SOURCE/'policy.pt',map_location='cpu',weights_only=False)
    policy = policy_from_native(source)
    actor = torch.jit.load(io.BytesIO(original['residual_actor_jit']),map_location='cpu').eval()
    with torch.inference_mode():
        expected = policy.act_inference(TensorDict({'policy':sample},batch_size=[len(sample)]))
        error = float((expected-actor(sample)).abs().max())
    assert error == 0., 'Rebuilt actor/normalizer differs from the actual completed native8 public actor'
    return dict(actual_public_samples=len(sample),actual_source8_actor_export_max_error=error,
        source_canonical8_sha256=sha(SOURCE/'checkpoint.pt'),source_public8_sha256=sha(SOURCE/'policy.pt'),
        native_restart_created=False,new_native_execution=False,
        sample_source_sha256={str(path.relative_to(ROOT)):sha(path) for path in sources})


def export(iteration):
    assert iteration in [16,24,32]
    directory = HERE/'exports'/f'iter{iteration:02d}'
    output = directory/'policy.pt'; receipt = directory/'EXPORT_CHECK.json'
    diagnostic = RUN/'ppo'/f'model_{iteration-1}.pt'
    if output.exists():
        checked = read(receipt)
        assert checked['policy_sha256'] == sha(output) and checked['source_diagnostic_sha256'] == sha(diagnostic)
        assert checked['configuration_sha256'] == sha(CONFIG) and checked['fixed_base_policy_sha256'] == sha(BASE)
        return output,checked
    assert not receipt.exists()
    # An actor at this boundary must have actually been collected afterward,
    # rather than marking an unexecuted newly proposed actor as native.
    progress = read(RUN/'progress.json')
    assert progress['update'] > iteration and progress['numerical_failures'] == 0
    inputs = read(RUN/'INPUTS.json')
    assert inputs['configuration_sha256'] == sha(CONFIG)
    assert inputs['contract']['base_policy_sha256'] == sha(BASE)
    assert inputs['contract']['base_controller_mode'] == 'frozen_public_policy'
    source = torch.load(SOURCE/'checkpoint.pt',map_location='cpu',weights_only=False)
    state = torch.load(diagnostic,map_location='cpu',weights_only=False)
    assert state['iter'] == iteration and not state['native']['canonical_restart']
    assert state['native']['training_signature'] == source['native']['training_signature']
    assert state['native']['environment']['num_envs'] == 2048
    assert state['native']['total_timesteps'] == iteration*2048*128
    assert len(state['optimizer_state_dict']['state']) == 17
    assert all(torch.isfinite(value).all() for part in state['optimizer_state_dict']['state'].values()
               for value in part.values() if torch.is_tensor(value))
    for name in ['actor_obs_normalizer.count','critic_obs_normalizer.count']:
        assert int(state['model_state_dict'][name]) == iteration*2048*128
    sample,sources = actual_samples(); policy = policy_from_native(state)
    base = torch.load(BASE,map_location='cpu',weights_only=False)
    result = export_residual_policy(base,policy,read(CONFIG)['joint_scale_rad'],sample,
        dict(native_execution=True,completed_updates=iteration,base_policy_sha256=sha(BASE),
            actual_control_env_steps=(iteration-8)*2048*128,base_controller_mode='frozen_public_policy',
            num_envs=2048,initialization=inputs['initialization'],source_diagnostic_sha256=sha(diagnostic),
            source_canonical_restart=False,CPU_export_of_actual_executed_native_actor=True,
            frozen_mean_policy_performance_inherited=False))
    assert result['base_controller']['residual_actor_jit'] == base['residual_actor_jit']
    assert not result['performance_inherited'] and not result['formal_acceptance']
    directory.mkdir(parents=True,exist_ok=False)
    temporary = output.with_suffix('.pt.tmp'); torch.save(result,temporary); temporary.replace(output)
    actor = torch.jit.load(io.BytesIO(result['residual_actor_jit']),map_location='cpu').eval()
    with torch.inference_mode():
        error = float((actor(sample)-policy.act_inference(TensorDict({'policy':sample},batch_size=[len(sample)]))).abs().max())
    assert error <= 2e-5
    checked = dict(iteration=iteration,source_diagnostic=str(diagnostic.relative_to(ROOT)),
        source_diagnostic_sha256=sha(diagnostic),policy_sha256=sha(output),
        configuration_sha256=sha(CONFIG),fixed_base_policy_sha256=sha(BASE),
        actual_public_samples=len(sample),actor_export_max_error=error,
        original_normalization_preserved=True,one_fixed_base_and_one_outer_residual=True,
        native_source_training_and_subsequent_collection_verified=True,
        CPU_export_only=True,new_native_execution=False,source_is_canonical_restart=False,
        scope='Actual executed native intermediate actor and public normalization frozen for later inference. Noncanonical diagnostic weights are not promoted to restart checkpoints or performance scores.',
        source_sha256={str(path.relative_to(ROOT)):sha(path) for path in
            [Path(__file__),diagnostic,CONFIG,BASE,RUN/'INPUTS.json',*sources]},
        formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    receipt.write_text(json.dumps(checked,indent=2)+'\n')
    return output,checked
