"""Verify actual full-first-cohort warmup7 followed by actor learning8."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = HERE/'ROOT_ACTUAL_FULL_COHORT_TO_ACTOR_CHECK.json'; assert not out.exists()
    spec = importlib.util.spec_from_file_location('actual_full_cohort_to_actor',HERE/'run.py')
    owner = importlib.util.module_from_spec(spec); spec.loader.exec_module(owner)
    assert owner.core.live(1777471,'warmup7_n2048_probe13_01')
    checks = [owner.verify_update(i) for i in [7,8]]
    a,b = [torch.load(owner.TRAIN/f'ppo/model_{i-1}.pt',map_location='cpu',weights_only=False) for i in [7,8]]
    assert a['native']['training_signature'] == b['native']['training_signature']
    assert checks[0]['mapped_actor_weights_unchanged'] and not checks[1]['mapped_actor_weights_unchanged']
    assert len(checks[0]['actual_Adam_steps']) == 8 and len(checks[1]['actual_Adam_steps']) == 17
    changed = [k for k in a['model_state_dict'] if k.startswith('actor.')
        and not torch.equal(a['model_state_dict'][k],b['model_state_dict'][k])]
    assert changed
    cohort = owner.core.read(HERE/'FIRST_COHORT_CHECK.json')
    assert cohort['first_episode_lanes'] == 2048 and cohort['all_actual_source_inclusive_episode_chains_exact']
    assert int(a['native']['environment']['tensors']['episode_index'].min()) >= 1
    plan = owner.core.read(HERE/'PLAN.json')
    assert all(sha(ROOT/name) == digest for name,digest in plan['source_hashes'].items())
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(),actual_live_native=1777471,
        actual_update7=checks[0],actual_update8=checks[1],first_ground_cohort=cohort['independently_recomputed_first_cohort'],
        actual_changed_actor_weight_tensors_at_first_actor_update=changed,
        actor_learning_began_after_all2048_initial_episodes_closed=True,main301_inputs_and_PLAN_unchanged=True,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),HERE/'FIRST_COHORT_CHECK.json',
            owner.TRAIN/'ppo/model_6.pt',owner.TRAIN/'ppo/model_7.pt',HERE/'PLAN.json']},
        scope='Actual learning transition. Normalization/exploration continue; first cohort is not frozen performance and no improvement is inferred.',
        new_native_execution=False,formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['actor_learning_began_after_all2048_initial_episodes_closed','first_ground_cohort']}),flush=True)


if __name__ == '__main__': main()
