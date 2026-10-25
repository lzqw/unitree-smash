"""Check pipeline boundaries against live owners and real native checkpoint data.

No simulation or policy learning is executed by this CPU check.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import torch

ROOT = Path(__file__).resolve().parents[3]
MAIN = ROOT.parent/'unitree_smash'
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = HERE/'ACTUAL_CPU_PIPELINE_CHECK.json'
    assert not output.exists()
    spec = importlib.util.spec_from_file_location('candidate075_workflow',HERE/'run.py')
    workflow = importlib.util.module_from_spec(spec);spec.loader.exec_module(workflow)
    observation = workflow.inspect_selection()
    assert observation['actual_original_owner_live'] and not observation['source_owner_and_ledger_clear']
    try:
        workflow.require_completed_selection()
    except AssertionError as error:
        assert 'still live' in str(error)
    else:
        raise AssertionError('Live source owner must prevent new preparation/launch')
    old = MAIN/'reports/G07_full_racket_v4/residual_ppo_centered_base8_n2048_probe8_02'
    first = torch.load(old/'ppo/model_0.pt',map_location='cpu',weights_only=False)
    final = torch.load(old/'checkpoint.pt',map_location='cpu',weights_only=False)
    first_steps, final_steps = workflow.optimizer_steps(first),workflow.optimizer_steps(final)
    assert first['iter'] == 1 and len(first_steps) == 8
    assert all(name.startswith('critic.') and step == 20 for name,step in first_steps.items())
    assert final['iter'] == 8 and len(final_steps) == 17
    assert sum(name.startswith('critic.') for name in final_steps) == 8
    assert sum(not name.startswith('critic.') for name in final_steps) == 9
    assert all(value == (160 if name.startswith('critic.') else 120) for name,value in final_steps.items())
    strict = MAIN/'reports/G07_full_racket_v4/residual_ppo_centered_base8_n2048_resume32_01/INPUTS.json'
    try:
        workflow.validate_training_inputs(json.loads(strict.read_text()))
    except AssertionError as error:
        assert 'never relabel' in str(error)
    else:
        raise AssertionError('Old strict-resume native job must not be accepted as the new075 stage')
    before = {name:(HERE/name).exists() for name in ['state.json','observer.lock','PREDECLARED_PLAN.json']}
    command = [str(Path(os.sys.executable)),str(HERE/'run.py'),'--prepare-only']
    attempted = subprocess.run(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2'),
                               capture_output=True,text=True,timeout=30)
    assert attempted.returncode != 0 and 'integrated original main worktree' in attempted.stderr
    assert before == {name:(HERE/name).exists() for name in before}
    native_identities = []
    for entry in observation['active_project_jobs']:
        for key in ['pid','child_pid']:
            path = Path('/proc')/str(entry[key])/'cmdline'
            if path.exists() and str(entry['output']).encode() in path.read_bytes():
                native_identities.append(dict(pid=entry[key],command=path.read_bytes().replace(b'\0',b' ').decode()))
    assert native_identities
    paths = [Path(__file__),HERE/'run.py',old/'ppo/model_0.pt',old/'checkpoint.pt',strict]
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(),scope='CPU workflow boundary checks using real existing native metadata; no new native execution.',
                  actual_original_owner_observation=observation,actual_guard_and_native_process_identities=native_identities,
                  live_source_owner_prevents_new_launch=True,candidate_local_ledger_execution_rejected=True,
                  no_plan_or_state_or_lock_created_by_rejected_candidate_preparation=True,
                  real_old_first_update_critic_Adam_states=8,real_old_first_update_critic_Adam_steps=20,
                  real_old_pilot8_actor_Adam_states=9,real_old_pilot8_actor_Adam_steps=120,
                  real_old_pilot8_critic_Adam_states=8,real_old_pilot8_critic_Adam_steps=160,
                  old_strict_resume_job_rejected_as_new075_training=True,
                  native_pipeline_status='PREPARED_NOT_INTEGRATED_NOT_RUN',
                  limitations='Old checkpoint data validate field/layout parsing only; they are not relabelled as a075 run or proof of the future native transfer.',
                  source_sha256={str(path):sha(path) for path in paths},
                  new_native_execution=False,new_training=False,formal_acceptance=False,goal_complete=False)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ['live_source_owner_prevents_new_launch','candidate_local_ledger_execution_rejected',
                      'real_old_first_update_critic_Adam_states','real_old_pilot8_actor_Adam_steps','old_strict_resume_job_rejected_as_new075_training','new_native_execution']}))


if __name__ == '__main__':
    main()
