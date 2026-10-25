"""Check new zero residual against the actual measured public base controller.

Saved native inputs exercise CPU command composition, including a real reset.
This does not run dynamics or inherit the source policy's physical results.
"""
import argparse
import copy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from rsl_rl.modules import ActorCritic
from tensordict import TensorDict

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from quadsmash.control.full_racket_residual_v4 import FullRacketResidualControllerV4
from quadsmash.rl.full_racket_residual_ppo_v4 import export_residual_policy
from quadsmash.rl.frozen_baseline_source_v4 import verify_frozen_baseline_source


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--base-checkpoint',type=Path)
    args=parser.parse_args();assert not args.output.exists()
    source=args.source_root.resolve();reports=source/'reports/G07_full_racket_v4'
    base_path=args.base_checkpoint or reports/'residual_ppo_joint_placement_actor32_n2048_probe8_01/policy.pt'
    base=torch.load(base_path,map_location='cpu',weights_only=False)
    protocol_path=ROOT/'configs/g07_full_racket_eval_v4_r3.json';protocol=read(protocol_path)
    provenance=verify_frozen_baseline_source(base_path,base,sha(protocol_path))
    for name,expected in {**protocol['physical_configs'],protocol['asset']['path']:protocol['asset']['sha256']}.items():
        assert sha(ROOT/name)==sha(source/name)==expected
    config_path=ROOT/'configs/g07_full_racket_residual_ppo_centered_v4.json';config=read(config_path)
    assert config['base_controller_mode']=='frozen_public_policy' and config['joint_scale_rad']==[.03]*12+[.15]*6
    run_names=['residual_ppo_joint_placement_actor32_n2048_probe8_fixed_train128_01',
        'residual_ppo_joint_placement_actor32_n2048_iter40_fixed_train128_01']
    streams=[];dependencies=[Path(__file__).resolve(),base_path,protocol_path,config_path]
    for name in run_names:
        run=reports/name;assert read(run/'job.json')['status']=='SUCCEEDED'
        data=np.load(run/'public_and_actions_first2.npz')
        streams.append({key:data[key]for key in data.files})
        dependencies.extend([run/'job.json',run/'public_and_actions_first2.npz'])
    public=torch.tensor(streams[0]['observations'].reshape(-1,794)[:200]);obs=TensorDict({'policy':public},batch_size=[200])
    settings=copy.deepcopy(config['runner']['policy']);settings.pop('class_name')
    torch.manual_seed(748701);policy=ActorCritic(obs,config['runner']['obs_groups'],18,**settings).eval()
    output=[m for m in policy.actor.modules()if isinstance(m,torch.nn.Linear)][-1]
    with torch.no_grad():output.weight.zero_();output.bias.zero_()
    optimizer=torch.optim.Adam(policy.parameters(),lr=config['runner']['algorithm']['learning_rate'])
    assert not optimizer.state
    exported=export_residual_policy(base,policy,config['joint_scale_rad'],public,
        dict(native_execution=False,base_controller_mode='frozen_public_policy',completed_updates=0,
            actual_control_env_steps=0,base_policy_sha256=sha(base_path)))
    assert not exported['native_execution'] and not exported['performance_inherited']
    with torch.inference_mode():assert policy.act_inference(obs).eq(0).all()
    authored=read(reports/'CPU_REPLAY_AUTHORED_JOINTS_0724.json')
    names=[f'{leg}_{joint}_joint'for joint in ['hip','thigh','calf']for leg in ['FL','FR','RL','RR']]+[f'joint{i}'for i in range(1,7)]
    defaults=dict(authored['defaults_before_neutral_arm_override'])
    defaults.update({f'joint{i+1}':q for i,q in enumerate(read(ROOT/'configs/full_racket_reset_v4.json')['arm_joint_positions_rad'])})
    default=torch.tensor([defaults[n]for n in names],dtype=torch.float32)
    limits=torch.tensor([authored['limits'][n]for n in names],dtype=torch.float32)
    checks=[]
    for name,stream in zip(run_names,streams):
        original=FullRacketResidualControllerV4(base,2,'cpu')
        centered=FullRacketResidualControllerV4(exported,2,'cpu')
        assert all(not p.requires_grad for p in centered.base.actor.parameters())
        last=None;resets=0;maximum_native_error=0.
        compare_native=(name==run_names[0] and base_path.name=='policy.pt' and 'probe8_01'in str(base_path))
        for tick,row in enumerate(stream['observations']):
            observed=torch.tensor(row);before=observed.clone()
            if last is not None:
                ids=torch.where(observed[:,775]<last-1e-6)[0]
                if len(ids):original.reset(ids);centered.reset(ids);resets+=len(ids)
            expected=original.action({'policy':observed},default,limits)
            actual=centered.action({'policy':observed},default,limits)
            assert torch.equal(actual,expected) and torch.equal(observed,before)
            assert torch.isfinite(actual).all() and actual.abs().max()<=1.
            if compare_native:
                valid=stream['valid_first_episode'][tick]
                if valid.any():maximum_native_error=max(maximum_native_error,float(np.abs(actual.numpy()-stream['actions'][tick])[valid].max()))
            last=observed[:,775].clone()
        checks.append(dict(actual_source_run=name,controls=len(stream['observations']),public_rows=int(np.prod(stream['observations'].shape[:2])),
            zero_new_residual_vs_fixed_base_bitwise_equal=True,actual_reset_events_replayed=resets,
            CPU_vs_original_native_command_comparison_performed=compare_native,
            CPU_vs_original_native_command_max_error=maximum_native_error if compare_native else None))
    assert sum(v['public_rows']for v in checks)==3200 and sum(v['actual_reset_events_replayed']for v in checks)>=1
    dependencies.extend([ROOT/'quadsmash/control/full_racket_residual_v4.py',ROOT/'quadsmash/rl/full_racket_residual_ppo_v4.py',
        ROOT/'quadsmash/rl/frozen_baseline_source_v4.py',ROOT/'scripts/train_full_racket_residual_ppo_v4.py'])
    result=dict(checked_at=datetime.now(timezone.utc).isoformat(),
        scope='CPU composition/export on3200 actual native public inputs, source execution receipt checks and fresh outer learner construction. Not a new native execution or policy-performance result.',
        baseline_source=provenance,config_sha256=sha(config_path),actual_saved_input_checks=checks,
        source_base_actor_and_normalization_fixed=True,outer_actor_critic_and_normalizers_fresh=True,
        outer_Adam_empty=True,outer_mean_residual_exactly_zero=True,outer_trainable_action_joints=18,
        leg_increment_bound_rad=.03,arm_increment_bound_rad=.15,original_native_actuator_limits_changed=False,
        outer_export_native_execution=False,new_native_training_run=False,previous_goal_turn_classification='progress: actual frozen40 collapse and first real feedback outcomes were independently checked and published; latest video/raw delivery completed.',
        next_native_action='After current original-source owners finish, select the measured base from complete TRAIN diagnostics; integrate and run real centered training with the full trajectory recorder under the main original GPU0 guard.',
        source_sha256={str(p):sha(p)for p in dependencies},working_directory=str(Path.cwd()),
        formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    args.output.parent.mkdir(parents=True,exist_ok=True);tmp=args.output.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(args.output)
    print(json.dumps(dict(actual_public_rows=3200,actual_reset_events=sum(v['actual_reset_events_replayed']for v in checks),
        zero_outer_residual_bitwise_equal=True,native_execution=False)),flush=True)


if __name__=='__main__':main()
