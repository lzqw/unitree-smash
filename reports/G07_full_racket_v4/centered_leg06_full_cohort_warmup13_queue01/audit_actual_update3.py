"""Compare real diagnostic3 at the first intended old/new warmup divergence."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REPORTS = ROOT/'reports/G07_full_racket_v4'
OLD = REPORTS/'residual_ppo_centered_leg06_after075_intermediates_n2048_probe8_01/ppo/model_2.pt'
NEW = REPORTS/'residual_ppo_centered_leg06_full_cohort_warmup7_n2048_probe13_01/ppo/model_2.pt'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = HERE/'ROOT_ACTUAL_UPDATE3_COMPARISON.json'; assert not output.exists()
    spec = importlib.util.spec_from_file_location('actual_warmup13_update3_validation',HERE/'run.py')
    owner = importlib.util.module_from_spec(spec); spec.loader.exec_module(owner)
    assert owner.core.live(1777471,'warmup7_n2048_probe13_01')
    checked = owner.verify_update(3)
    old,new = [torch.load(p,map_location='cpu',weights_only=False) for p in [OLD,NEW]]
    assert old['iter'] == new['iter'] == 3
    assert old['native']['total_timesteps'] == new['native']['total_timesteps'] == 786432
    names = [name for name in new['model_state_dict'] if name.startswith('actor.')]
    assert len(names) == 8
    changed = [name for name in names if not torch.equal(old['model_state_dict'][name],new['model_state_dict'][name])]
    assert changed
    old_steps = owner.reference.optimizer_steps(old)
    assert len(old_steps) == 17 and all(v == (60 if k.startswith('critic.') else 20) for k,v in old_steps.items())
    assert len(checked['actual_Adam_steps']) == 8 and all(v == 60 for v in checked['actual_Adam_steps'].values())
    physical = {}
    for name in ['root','joint_position','joint_velocity']:
        a,b = old['native']['environment'][name],new['native']['environment'][name]
        assert a.shape == b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
        physical[name] = dict(bitwise_equal=torch.equal(a,b),max_abs_difference=float((a-b).abs().max()))
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(),actual_live_native=1777471,
        actual_current_job_steps_at_diagnostic=786432,new_critic_only_update3=checked,
        old_actor_Adam_steps20=True,old_critic_Adam_steps60=True,
        old_vs_new_actor_weight_tensors_different=changed,
        same_seed_first3_rollouts_terminal_physical_comparison=physical,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),HERE/'run.py',OLD,NEW]},
        scope='Actual two completed diagnostic3 states. Validates executed warmup difference, not improved task performance or a general bitwise GPU reproducibility guarantee.',
        new_native_execution=False,formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['actual_current_job_steps_at_diagnostic','old_actor_Adam_steps20',
        'same_seed_first3_rollouts_terminal_physical_comparison']}),flush=True)


if __name__ == '__main__': main()
