from pathlib import Path
from datetime import datetime,timezone
import contextlib,hashlib,io,json,sys
import numpy as np
import torch
from rsl_rl.modules import ActorCritic
from tensordict import TensorDict
from quadsmash.rl.frozen_baseline_source_v4 import verify_frozen_baseline_source
from scripts.initialize_centered_policy_v4 import verify_measured_source
from scripts.initialize_centered_action_units_v4 import apply_action_unit_mapped_actor
root=Path.cwd(); reports=root/'reports/G07_full_racket_v4'
train=reports/'residual_ppo_centered_leg06_after075_intermediates_n2048_probe8_01'
source=reports/'residual_ppo_centered_placement075_n2048_probe8_01'
evaluation=reports/'residual_ppo_centered_placement075_n2048_probe8_fixed_train128_01'
base_path=reports/'residual_ppo_joint_placement_actor32_n2048_probe8_01/policy.pt'
base=torch.load(base_path,map_location='cpu',weights_only=False)
protocol=root/'configs/g07_full_racket_eval_v4_r3.json';digest=hashlib.sha256(protocol.read_bytes()).hexdigest()
base_provenance=verify_frozen_baseline_source(base_path,base,digest)
contract=json.loads((train/'INPUTS.json').read_text())['contract']
source_provenance=verify_measured_source(root,source/'policy.pt',evaluation,contract)
configuration=json.loads((root/'configs/g07_full_racket_residual_ppo_centered_leg06_placement075_v4.json').read_text())
options=dict(configuration['runner']['policy']);options.pop('class_name')
with contextlib.redirect_stdout(io.StringIO()):
 policy=ActorCritic(TensorDict({'policy':torch.zeros(2,794)},batch_size=[2]),configuration['runner']['obs_groups'],18,**options)
with np.load(evaluation/'public_and_actions_first2.npz') as a:witness=torch.from_numpy(a['observations'][a['valid_first_episode'].astype(bool)].copy())
canonical=torch.load(source/'checkpoint.pt',map_location='cpu',weights_only=False)
public=torch.load(source/'policy.pt',map_location='cpu',weights_only=False)
transfer=apply_action_unit_mapped_actor(policy,public,canonical['native']['interface'],contract,base,witness)
assert transfer['actual_witness_physical_mean_max_error_rad']<=2e-6
result=dict(checked_at=datetime.now(timezone.utc).isoformat(),root=str(root),baseline_source_receipts_verified=base_provenance,measured_source_receipts_verified=source_provenance,transfer=transfer,source_provenance_and_initialization_passed=True,software_external=True,native_replay_performed=False,formal_cases_read=False,formal_acceptance=False,goal_complete=False)
Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+'\n');print('Combined archive source and initialization checks passed')
