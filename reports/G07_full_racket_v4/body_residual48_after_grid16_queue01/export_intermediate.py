"""Freeze real trained third-residual intermediates for inference only."""
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

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from quadsmash.rl.full_racket_body_residual_ppo_v4 import export_residual_policy
from scripts.initialize_centered_policy_v4 import same_tree

def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def from_state(state,config):
    options=dict(config['runner']['policy']);options.pop('class_name')
    with contextlib.redirect_stdout(io.StringIO()):
        policy=ActorCritic(TensorDict({'policy':torch.zeros(2,794)},batch_size=[2]),
            config['runner']['obs_groups'],18,**options)
    policy.load_state_dict(state['model_state_dict'],strict=True);policy.eval()
    assert all(torch.isfinite(v).all() for v in policy.state_dict().values() if torch.is_tensor(v))
    return policy

def export(iteration,run,base_path,evaluation):
    assert iteration in [13,24,36]
    directory=HERE/'exports'/f'iter{iteration:02d}';assert not directory.exists()
    diagnostic=run/'ppo'/f'model_{iteration-1}.pt'
    inputs=read(run/'INPUTS.json');config_path=ROOT/'configs/g07_full_racket_body_residual_leg15_v4.json'
    config=read(config_path)
    assert inputs['configuration_sha256']==sha(config_path)
    assert inputs['contract']['base_policy_sha256']==sha(base_path)
    # The complete native run must have subsequently collected this actor.
    assert read(run/'job.json')['status']=='SUCCEEDED'
    assert read(run/'progress.json')['update']>iteration
    state=torch.load(diagnostic,map_location='cpu',weights_only=False)
    assert state['iter']==iteration and not state['native']['canonical_restart']
    assert state['native']['total_timesteps']==iteration*2048*128
    assert state['native']['environment']['num_envs']==2048
    assert len(state['optimizer_state_dict']['state'])==17
    assert all(torch.isfinite(v).all() for s in state['optimizer_state_dict']['state'].values()
        for v in s.values() if torch.is_tensor(v))
    for name in ['actor_obs_normalizer.count','critic_obs_normalizer.count']:
        assert int(state['model_state_dict'][name])==iteration*2048*128
    source=evaluation/'public_and_actions_first2.npz'
    with np.load(source) as data:sample=torch.from_numpy(data['observations'][data['valid_first_episode']].copy())
    assert sample.shape[1]==794 and torch.isfinite(sample).all()
    policy=from_state(state,config);base=torch.load(base_path,map_location='cpu',weights_only=False)
    exported=export_residual_policy(base,policy,config['joint_scale_rad'],sample,
        dict(native_execution=True,completed_updates=iteration,actual_control_env_steps=iteration*2048*128,
            base_policy_sha256=sha(base_path),base_controller_mode='frozen_public_policy',num_envs=2048,
            initialization=inputs['initialization'],source_diagnostic_sha256=sha(diagnostic),
            source_canonical_restart=False,CPU_export_of_actual_subsequently_collected_actor=True))
    assert same_tree(exported['base_controller'],base)
    assert not exported['performance_inherited'] and not exported['formal_acceptance']
    actor=torch.jit.load(io.BytesIO(exported['residual_actor_jit']),map_location='cpu').eval()
    with torch.inference_mode():
        expected=policy.act_inference(TensorDict({'policy':sample},batch_size=[len(sample)]))
        maximum=float((expected-actor(sample)).abs().max())
    assert maximum<=2e-5
    directory.mkdir(parents=True,exist_ok=False);output=directory/'policy.pt';torch.save(exported,output)
    receipt=dict(iteration=iteration,policy_sha256=sha(output),source_diagnostic_sha256=sha(diagnostic),
        source_is_canonical_restart=False,new_native_execution=False,actual_public_input_rows=len(sample),
        actor_export_max_error=maximum,fixed_body_teacher_and_two_source_actors_exactly_preserved=True,
        source_sha256={str(p):sha(p) for p in [Path(__file__),config_path,base_path,source,diagnostic,run/'INPUTS.json']},
        formal_acceptance=False)
    (directory/'EXPORT_CHECK.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return output,receipt
