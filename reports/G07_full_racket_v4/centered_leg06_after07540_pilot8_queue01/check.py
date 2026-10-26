"""CPU workflow boundaries on real075 jobs and measured8/40 witnesses.

Old075 tensors exercise parser layouts only; they are explicitly rejected as
new leg-unit training. No candidate native execution is performed.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess

import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
MAIN = ROOT.parent/'unitree_smash'
PYTHON = '/home/autodrive/Isaac/env_isaaclab/bin/python'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())


def main():
    output = HERE/'ACTUAL_CPU_PIPELINE_CHECK.json';assert not output.exists()
    spec = importlib.util.spec_from_file_location('candidate_leg06_queue',HERE/'run.py')
    queue = importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)
    observed = queue.inspect_selection()
    assert observed['actual_original_owner_live'] and not observed['source_owner_and_ledger_clear']
    assert not observed['main_worktree']
    try:
        queue.require_completed_selection()
    except AssertionError as error:
        assert 'still live' in str(error)
    else:raise AssertionError('Actual075 owner must block selection')
    outputs = [HERE/name for name in ['PREDECLARED_PLAN.json','state.json','SOURCE_SELECTION.json',
        'SELECTED_SOURCE_CPU_ACTION_UNIT_CHECK.json','observer.lock']]
    assert not any(p.exists() for p in outputs)
    result = subprocess.run([PYTHON,str(HERE/'run.py'),'--prepare-only'],cwd=ROOT,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2'),text=True,capture_output=True)
    assert result.returncode != 0 and 'integrated original main worktree' in result.stderr
    assert not any(p.exists() for p in outputs)
    reports = MAIN/'reports/G07_full_racket_v4'
    prior = reports/'residual_ppo_centered_placement075_n2048_probe8_01'
    first = torch.load(prior/'ppo/model_0.pt',map_location='cpu',weights_only=False)
    final = torch.load(prior/'checkpoint.pt',map_location='cpu',weights_only=False)
    first_steps,final_steps = queue.optimizer_steps(first),queue.optimizer_steps(final)
    assert len(first_steps) == 8 and all(k.startswith('critic.') and v == 20 for k,v in first_steps.items())
    assert len(final_steps) == 17 and all(v == (160 if k.startswith('critic.') else 120) for k,v in final_steps.items())
    try:
        queue.validate_training_inputs(read(prior/'INPUTS.json'))
    except AssertionError as error:
        assert 'never relabel075' in str(error)
    else:raise AssertionError('Actual old075 inputs cannot stand for new action-unit execution')
    current = reports/'residual_ppo_centered_placement075_n2048_resume32_02'
    assert not (reports/'residual_ppo_centered_placement075_n2048_iter40_fixed_train128_02/metrics.json').exists()
    try:
        queue.validate_training_inputs(read(current/'INPUTS.json'))
    except AssertionError as error:
        assert 'never relabel075' in str(error)
    else:raise AssertionError('Actual strict075 continuation is not fresh leg-unit training')
    witnesses = []
    for iteration in [8,40,75]:
        folder = ('residual_ppo_centered_placement075_n2048_probe8_01' if iteration == 75 else 'residual_ppo_centered_base8_n2048_probe8_02' if iteration == 8 else 'residual_ppo_centered_base8_n2048_resume32_01')
        selected = dict(candidate_id=('CPU_witness_placement075_stage8' if iteration == 75 else f'CPU_witness_centered{iteration}'),policy=str(reports/folder/'policy.pt'),
            evaluation=str(reports/'residual_ppo_centered_placement075_n2048_probe8_fixed_train128_01' if iteration == 75 else queue.source_evaluation(iteration)))
        cpu = queue.selected_source_cpu_check(selected)
        assert cpu['transfer']['actual_witness_physical_mean_max_error_rad'] <= 2e-6 and cpu['fresh_Adam_entries'] == 0
        assert cpu['transfer']['source_actor_normalizer_count'] == (12582912 if iteration == 75 else 2097152 if iteration == 8 else 10485760)
        mapped = queue.mapped_initial_actor(selected)
        public = torch.load(selected['policy'],map_location='cpu',weights_only=False)
        actor = torch.jit.load(io.BytesIO(public['residual_actor_jit']),map_location='cpu').state_dict()
        assert torch.equal(mapped['actor.6.weight'][:12],actor['1.6.weight'][:12]*.5)
        assert torch.equal(mapped['actor.6.bias'][:12],actor['1.6.bias'][:12]*.5)
        assert torch.equal(mapped['actor.6.weight'][12:],actor['1.6.weight'][12:])
        assert any(not torch.equal(mapped['actor.'+name[2:]],v) for name,v in actor.items() if name.startswith('1.'))
        witnesses.append(cpu)
    active_plan = reports/'centered_placement075_resume32_queue02/PREDECLARED_PLAN.json'
    held = dict(read(active_plan)['source_hashes']);held[str(active_plan.relative_to(MAIN))] = sha(active_plan)
    for name,digest in held.items():
        path = Path(name) if Path(name).is_absolute() else MAIN/name
        assert sha(path) == digest,name
    record = dict(checked_at=datetime.now(timezone.utc).isoformat(),actual_source_observation=observed,
        candidate_prepare_rejected_before_creating_plan_state_lock=True,actual_live_owner_blocks_selection=True,
        actual_old075_first_optimizer_layout='8critic states each20',actual_old075_final_optimizer_layout='9actor/logstd120,8critic160',
        actual_old075_inputs_rejected_as_new_action_unit_stage=True,actual_strict075_inputs_rejected_as_new_action_unit_stage=True,
        actual07540_frozen_results_still_absent=True,mapped_initial_actor_checked_instead_of_raw_source=True,
        actual_measured_source_CPU_witnesses=witnesses,witnesses_do_not_select_a_future_source=True,
        all_main_held_sources_unchanged=len(held),workflow_sha256=sha(HERE/'run.py'),checker_sha256=sha(Path(__file__)),
        source_native_boundary_sha256={'old075_model0':sha(prior/'ppo/model_0.pt'),'old075_canonical8':sha(prior/'checkpoint.pt')},
        new_native_execution=False,new_training=False,formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    with output.open('x') as file:json.dump(record,file,indent=2);file.write('\n')
    print(json.dumps({key:record[key] for key in ['candidate_prepare_rejected_before_creating_plan_state_lock',
        'actual_live_owner_blocks_selection','actual_old075_inputs_rejected_as_new_action_unit_stage',
        'all_main_held_sources_unchanged','new_native_execution']}))


if __name__ == '__main__':main()
