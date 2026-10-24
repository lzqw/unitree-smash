"""Check causal preparation on saved public TRAIN observations, not new dynamics."""
import copy,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from quadsmash.control.full_racket_rally_controller_v4 import FullRacketRallyControllerV4
PRIMARY=Path('/home/autodrive/LZQW/unitree_smash');R=PRIMARY/'reports/G07_full_racket_v4';OUT=Path(__file__).parent
assert not (OUT/'REPORT.json').exists();torch.set_num_threads(2)
basepath=R/'slew50_near_attack_gen0_c54_1540/policy_evidence.pt';base=torch.load(basepath,map_location='cpu',weights_only=True)
warmpath=ROOT/'reports/G07_full_racket_v4/relative_prepose38_ready04_1615/policy.pt';enabled=torch.load(warmpath,map_location='cpu',weights_only=True)
contractpath=R/'CPU_REPLAY_AUTHORED_JOINTS_0724.json';contract=json.loads(contractpath.read_text())
names=[f'{leg}_{joint}_joint' for joint in ['hip','thigh','calf'] for leg in ['FL','FR','RL','RR']]+[f'joint{i}' for i in range(1,7)]
defaults=contract['defaults_before_neutral_arm_override'];reset=json.loads((ROOT/'configs/full_racket_reset_v4.json').read_text())['arm_joint_positions_rad'];defaults.update({f'joint{i+1}':q for i,q in enumerate(reset)})
default=torch.tensor([defaults[n] for n in names],dtype=torch.float32);limits=torch.tensor([contract['limits'][n] for n in names],dtype=torch.float32)
zero=copy.deepcopy(enabled);zero['seed']['causal_arm_prepose_gain']=0.
controllers=[FullRacketRallyControllerV4(c,2,'cpu') for c in [base,zero,enabled]]
for c in controllers:c.reset()
publicpath=R/'ball_arm_target_slew45_normal38_cem22/public_and_poses_000.npz'
count=0;changes=0
with np.load(publicpath) as saved,torch.inference_mode():
 for index,obs in enumerate(saved['observations']):
  if not saved['valid_first_episode'][index].all():break
  observation={'policy':torch.tensor(obs)};actions=[c.action(observation,default,limits) for c in controllers]
  assert torch.equal(actions[0],actions[1]),'Zero gain must preserve every default command exactly'
  assert torch.equal(actions[0][:,:12],actions[2][:,:12]),'On identical public inputs only arm preparation may change'
  assert all(torch.isfinite(a).all() and (abs(a)<=1).all() for a in actions)
  changes+=int(not torch.equal(actions[0][:,12:],actions[2][:,12:]));count+=1
assert count==506 and changes>0 and all(c.planned.all() for c in controllers)
planspath=R/'ball_arm_target_slew50_normal38_cem23/causal_plans_000.json';plan=json.loads(planspath.read_text())[0]
ids=[109,108];public=np.array([plan['used_public_observations'][plan['env_ids'].index(i)] for i in ids],dtype=np.float32)
anchor=FullRacketRallyControllerV4(enabled,2,'cpu');anchor.reset();anchor.action({'policy':torch.tensor(public)},default,limits)
assert torch.equal(anchor.parameters[0],base['parameters']),'Actual measured anchor must preserve parameters, including raw joint6offset above action bound'
assert not torch.equal(anchor.parameters[1],base['parameters']),'The paired different incoming forecast must produce an adaptation'
delta=(anchor.parameters[:,8:14]-base['parameters'][None,8:14]).numpy()
assert abs(delta).max()<=.30+1e-6
oldcommands=R/'servo_contact38_bank_cpu01_1600/counterfactual_commands.npz';newcommands=ROOT/'reports/G07_full_racket_v4/relative_prepose_legacy_bank_cpu01_1605/counterfactual_commands.npz'
with np.load(oldcommands) as old,np.load(newcommands) as new:
 np.testing.assert_array_equal(old['actions'],new['actions']);shape=list(new['actions'].shape)
sources=[basepath,warmpath,contractpath,publicpath,planspath,oldcommands,newcommands,Path(__file__),ROOT/'quadsmash/control/full_racket_rally_controller_v4.py',ROOT/'quadsmash/control/full_racket_relative_prepose_v4.py']
report=dict(verified_at=datetime.now(timezone.utc).isoformat(),scope='CPU public-input replay only. No changed-state simulation, new contact, recovery or success is proven.',
 default_none_legacy_bank_commands_bitwise_equal=True,legacy_bank_shape=shape,
 zero_gain_all_commands_bitwise_equal=True,zero_gain_public_ticks=count,
 enabled_legs_on_same_inputs_bitwise_equal=True,enabled_arm_command_changed_ticks=changes,
 actual_TRAIN65_reference_all38parameters_bitwise_preserved=True,
 paired_TRAIN1_forecast_produces_bounded_adaptation=True,anchor_and_paired_joint_delta_rad=delta.tolist(),
 public_forecast_only_no_case_id_runtime_input=True,native_execution=False,formal_acceptance=False,
 source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
(OUT/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))
