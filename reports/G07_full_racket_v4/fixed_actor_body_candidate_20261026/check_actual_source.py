"""CPU identity check using a real completed source and actual public streams."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
MAIN = ROOT.parent/'unitree_smash'
HERE = Path(__file__).resolve().parent
REPORTS = MAIN/'reports/G07_full_racket_v4'
SOURCE = REPORTS/'residual_ppo_centered_placement075_n2048_probe8_01'
EVAL = REPORTS/'residual_ppo_centered_placement075_n2048_probe8_fixed_train128_01'
sys.path.insert(0,str(ROOT))
from quadsmash.control.full_racket_fixed_actor_body_v4 import FixedActorBodyControllerV4
from quadsmash.control.full_racket_residual_v4 import make_rally_controller


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = HERE/'ACTUAL_SOURCE_IDENTITY_CHECK.json'; assert not out.exists()
    checkpoint = torch.load(SOURCE/'policy.pt',map_location='cpu',weights_only=False)
    assert read(EVAL/'metrics.json')['policy_sha256'] == sha(SOURCE/'policy.pt')
    assert read(EVAL/'job.json')['status'] == 'SUCCEEDED'
    provenance = read(SOURCE/'trajectory_traces/manifest.json')['provenance']
    default = torch.tensor(provenance['default_joint_position_rad'],dtype=torch.float32)
    limits = torch.tensor(provenance['joint_limits_rad'],dtype=torch.float32)
    with np.load(EVAL/'public_and_actions_first2.npz') as arrays:
        public = arrays['observations'].copy();valid = arrays['valid_first_episode'].copy()
    assert public.shape == (800,2,794) and valid.all()
    original_parameters = checkpoint['parameters'].clone()
    reference = make_rally_controller(checkpoint,2,'cpu')
    candidate = FixedActorBodyControllerV4(checkpoint,checkpoint['parameters'],2,'cpu')
    reference.reset();candidate.reset();error = 0.
    with torch.inference_mode():
        for sample in public:
            obs = {'policy':torch.from_numpy(sample)}
            a,b = reference.action(obs,default,limits),candidate.action(obs,default,limits)
            error = max(error,float((a-b).abs().max()))
    assert error == 0. and torch.equal(checkpoint['parameters'],original_parameters)
    actor_count = 0; node = candidate.source
    while hasattr(node,'actor'):
        assert all(not p.requires_grad for p in node.actor.parameters())
        actor_count += 1;node = node.base
    assert actor_count == 2
    candidate.reset();reference.reset()
    population = original_parameters[None].repeat(2,1)
    population[:,1] += .15;population[:,24] += .1
    population[:,37] = torch.minimum(population[:,37]+.02,torch.tensor(.12))
    candidate.set_parameters(population)
    assert torch.equal(candidate.reference.parameters,population)
    with torch.inference_mode():
        obs = {'policy':torch.from_numpy(public[201])}
        altered = candidate.action(obs,default,limits);original = reference.action(obs,default,limits)
    assert not torch.equal(altered,original)
    try: candidate.set_parameters(population)
    except ValueError: late_change_rejected = True
    else: raise AssertionError('A planned controller accepted an unreset population change')
    candidate.reset(torch.tensor([1]));assert torch.equal(candidate.reference.parameters[1],population[1])
    assert torch.equal(checkpoint['parameters'],original_parameters)
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(),actual_source_policy_sha256=sha(SOURCE/'policy.pt'),
        actual_public_rows=1600,full800step_source_identity_max_action_error=error,
        both_original_residual_actors_preserved_and_frozen=True,parameter_patch_changes_commands=True,
        per_lane_reset_keeps_declared_parameter_override=True,late_unreset_change_rejected=late_change_rejected,
        original_source_parameters_not_mutated=True,
        source_sha256={str(p):sha(p) for p in [Path(__file__),ROOT/'quadsmash/control/full_racket_fixed_actor_body_v4.py',
            SOURCE/'policy.pt',EVAL/'public_and_actions_first2.npz',SOURCE/'trajectory_traces/manifest.json']},
        scope='CPU compatibility/new-family preparation only. No altered policy executed on native robot; source performance not inherited.',
        new_native_execution=False,formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in
        ['actual_public_rows','full800step_source_identity_max_action_error','both_original_residual_actors_preserved_and_frozen']}),flush=True)


if __name__ == '__main__': main()
