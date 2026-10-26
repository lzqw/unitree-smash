"""Actual-source zero-residual compatibility and fresh learner checks, CPU only."""
import contextlib
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import py_compile
import sys
import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.modules import ActorCritic

ROOT=Path(__file__).resolve().parents[3]
MAIN=ROOT.parent/'unitree_smash'
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from quadsmash.control.full_racket_fixed_actor_body_v4 import make_rally_controller as make_body
from quadsmash.control.full_racket_body_residual_v4 import make_rally_controller
from quadsmash.rl.full_racket_body_residual_ppo_v4 import export_residual_policy
from quadsmash.rl.frozen_body_source_v4 import verify_frozen_body_source
from scripts.initialize_fresh_body_residual_v4 import initialize_fresh_body_actor


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())


def main():
    out=HERE/'ACTUAL_ZERO_BASE_COMPATIBILITY.json';assert not out.exists()
    base_path=MAIN/'reports/G07_full_racket_v4/body_candidate4_cold_queue01/candidate4_measured_public.pt'
    evaluation=MAIN/'reports/G07_full_racket_v4/fixed_actor_body_gen1_candidate4_fixed_train128_01'
    base=torch.load(base_path,map_location='cpu',weights_only=False)
    protocol=MAIN/'configs/g07_full_racket_eval_v4_r3.json'
    source=verify_frozen_body_source(base_path,base,evaluation,MAIN,sha(protocol))
    assert source['source_original_TRAIN128_summary']['complete']==1
    config_path=ROOT/'configs/g07_full_racket_body_residual_leg15_v4.json';config=read(config_path)
    options=dict(config['runner']['policy']);options.pop('class_name');torch.manual_seed(750301)
    with contextlib.redirect_stdout(io.StringIO()):
        policy=ActorCritic(TensorDict({'policy':torch.zeros(2,794)},batch_size=[2]),
            config['runner']['obs_groups'],18,**options)
    optimizer=torch.optim.Adam(policy.parameters(),lr=1e-4)
    initialized=initialize_fresh_body_actor(policy,config)
    with np.load(evaluation/'public_and_actions_first2.npz') as arrays:
        public=arrays['observations'].copy();valid=arrays['valid_first_episode'].astype(bool).copy()
    witness=torch.from_numpy(public[valid]);assert witness.shape[1]==794
    assert not optimizer.state and int(policy.actor_obs_normalizer.count)==int(policy.critic_obs_normalizer.count)==0
    exported=export_residual_policy(base,policy,config['joint_scale_rad'],witness[:16],
        dict(native_execution=False,completed_updates=0,actual_control_env_steps=0,
            base_policy_sha256=sha(base_path),base_controller_mode='frozen_public_policy'))
    assert not exported['native_execution'] and not exported['performance_inherited']
    original=make_body(base,2,'cpu');new=make_rally_controller(exported,2,'cpu')
    provenance=read(MAIN/'reports/G07_full_racket_v4/residual_ppo_centered_placement075_n2048_probe8_01/trajectory_traces/manifest.json')['provenance']
    default=torch.tensor(provenance['default_joint_position_rad']);limits=torch.tensor(provenance['joint_limits_rad'])
    original.reset();new.reset();previous=None;maximum=0.
    for sample,keep in zip(public,valid):
        tensor=torch.from_numpy(sample)
        if previous is not None:
            ids=torch.where(tensor[:,775]<previous-1e-5)[0]
            if len(ids):original.reset(ids);new.reset(ids)
        a=original.action({'policy':tensor},default,limits);b=new.action({'policy':tensor},default,limits)
        if keep.any():maximum=max(maximum,float((a[keep]-b[keep]).abs().max()))
        previous=tensor[:,775].clone()
    assert maximum==0.
    node=new.base.source;fixed=0
    while hasattr(node,'actor'):
        assert all(not parameter.requires_grad for parameter in node.actor.parameters());fixed+=1;node=node.base
    assert fixed==2
    paths=[Path(__file__),config_path,ROOT/'scripts/initialize_fresh_body_residual_v4.py',
        ROOT/'scripts/train_full_racket_body_residual_ppo_v4.py',ROOT/'scripts/evaluate_full_racket_body_residual_train_v4.py',
        ROOT/'quadsmash/control/full_racket_body_residual_v4.py',ROOT/'quadsmash/rl/full_racket_body_residual_ppo_v4.py',
        ROOT/'quadsmash/rl/frozen_body_source_v4.py',base_path,evaluation/'public_and_actions_first2.npz']
    for path in paths:
        if path.suffix=='.py':py_compile.compile(str(path),doraise=True)
    old=(MAIN/'scripts/evaluate_full_racket_train_v4.py').read_text()
    assert (ROOT/'scripts/evaluate_full_racket_body_residual_train_v4.py').read_text()==old.replace(
        'from quadsmash.control.full_racket_residual_v4 import make_rally_controller',
        'from quadsmash.control.full_racket_body_residual_v4 import make_rally_controller')
    data=dict(checked_at=datetime.now(timezone.utc).isoformat(),source_sha256={str(p):sha(p) for p in paths},
        actual_source=source,initialization=initialized,actual_valid_source_public_rows=int(valid.sum()),
        zero_residual_source_action_max_error=maximum,both_existing_actors_preserved_and_frozen=True,
        new_trainable_actor_count=1,total_residual_actor_count=3,source_locomotion_and_strike_backbones_preserved=True,new_Adam_entries=0,
        new_actor_and_critic_RMS_counts=[0,0],cold_evaluation_only_factory_import_changed=True,
        new_policy_native_execution=False,formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    out.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(dict(actual_valid_source_public_rows=int(valid.sum()),zero_residual_error=maximum,
        fixed_actor_count=2,new_actor_count=1,source_cold128=source['source_original_TRAIN128_summary']['complete'],
        fresh_leg_physical_std=.03,new_native_execution=False)),flush=True)


if __name__=='__main__':main()
