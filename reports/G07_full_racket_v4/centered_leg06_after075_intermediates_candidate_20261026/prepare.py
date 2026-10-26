"""Declare optional ten-source leg-unit pilot before frozen075 intermediates."""
from datetime import datetime, timezone
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
MAIN = ROOT.parent/'unitree_smash'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = HERE/'CONDITIONAL_PLAN.json'
    assert not output.exists()
    reports = MAIN/'reports/G07_full_racket_v4'
    comparison = reports/'centered_placement075_intermediate_selection_queue01'
    owner = reports/'centered_placement075_resume32_queue02'
    for pid, needle in [(1734658, 'centered_placement075_resume32_queue02/run.py'),
        (1734680, 'residual_ppo_centered_placement075_n2048_resume32_02'),
        (1744903, 'centered_placement075_intermediate_selection_queue01/run.py')]:
        assert needle.encode() in Path('/proc', str(pid), 'cmdline').read_bytes()
    pending = [dict(candidate_id=f'placement075_stage{i}', iteration=i,
        policy=str(comparison/'exports'/f'iter{i:02d}'/'policy.pt'),
        evaluation=str(reports/f'residual_ppo_centered_placement075_diag_iter{i}_fixed_train128_01'))
        for i in [16, 24, 32]]
    pending.append(dict(candidate_id='placement075_stage40', iteration=40,
        policy=str(reports/'residual_ppo_centered_placement075_n2048_resume32_02/policy.pt'),
        evaluation=str(reports/'residual_ppo_centered_placement075_n2048_iter40_fixed_train128_02')))
    assert all(not (Path(e['evaluation'])/'metrics.json').exists() for e in pending)
    held = dict(read(comparison/'PREDECLARED_PLAN.json')['source_hashes'])
    held[str((comparison/'PREDECLARED_PLAN.json').relative_to(MAIN))] = sha(comparison/'PREDECLARED_PLAN.json')
    assert all(sha(MAIN/name) == digest for name, digest in held.items())
    workflow = HERE.parent/'centered_leg06_after075_intermediates_pilot8_queue01'
    check = read(workflow/'ACTUAL_CPU_PIPELINE_CHECK.json')
    assert check['actual_live_owner_blocks_selection'] and check['actual_live_intermediate_owner_blocks_selection']
    assert check['CPU_admission_scenario_prior_owner_and_ledger_clear_but_real_intermediate_observer_live_blocks']
    assert check['candidate_prepare_rejected_before_creating_plan_state_lock']
    assert check['actual07516_24_32_frozen_results_still_absent']
    old_path = HERE.parent/'centered_leg06_after07540_candidate_20261026/CONDITIONAL_PLAN.json'
    old = read(old_path)
    new = copy.deepcopy(old)
    progress = read(reports/'residual_ppo_centered_placement075_n2048_resume32_02/progress.json')
    analysis = reports/'placement075_action_authority_20261026/ACTUAL_SELECTED_TRACE_ACTION_CHECK.json'
    action = read(analysis)
    assert action['selected_actual_traces'] == 15 and action['all_selected_trace_samples'] == 5836
    assert max(e['CPU_mixer_vs_saved_native_action_max_error'] for e in action['episodes']) == 0.
    new.update(created_at=datetime.now(timezone.utc).isoformat(),
        status='OPTIONAL_CPU_PREPARED_NOT_SELECTED_NOT_MAIN_INTEGRATED_NOT_NATIVE_RUN',
        prior_declaration_preserved=str(old_path), prior_declaration_sha256=sha(old_path),
        decision_timing='After actual075 frozen8, three audited exploratory TRAIN successes, zero-error1366-input native16 CPU export, and5836-row exact command reconstruction; before any075 frozen16/24/32 or40 result.',
        actual_current_TRAIN_update_at_registration=progress['update'],
        actual_current_TRAIN_complete_count_at_registration=progress['complete'],
        hypothesis='Change leg action units and the physical response to learned output updates, preserving observed source physical mean and reference fresh exploration. Actual telemetry does not show the .03rad residual cap binding. A larger cap is not a proven remedy; measure the new pilot before extending it.',
        evidence_scope='Original075 frozen8 is0complete/26air/19recoveries. Exploratory TRAIN successes are not inherited. Exact command reconstruction rules out silent residual omission but is not a causal native ablation.',
        main_held_sources_unchanged=len(held))
    select = new['source_selection_after_actual075_completion']
    known = select['known_original_candidates']
    assert len(known) == 6
    select['candidate_order_for_final_tie'] = [e['candidate_id'] for e in known]+[e['candidate_id'] for e in pending]
    select.pop('pending_candidate')
    select['pending_candidates'] = pending
    select['required'] = 'Both actual1734658 train/fixed/video and1744903 intermediate observers exit successfully, original main ledger clear. Recompute all ten original TRAIN128 sources; missing intermediates allowed only when comparison RESULT explicitly skipped them because a measured source was already50/70/85/NF0-ready. Any ready source skips this optional revision for original formal review.'
    new['launch_requirements'] = [
        'Actual075 train/fixed/video observer1734658 exits READY_FOR_VISUAL_REVIEW and comparison observer1744903 exits COMPLETED_INTERMEDIATE_SELECTION. No main RUNNING job.',
        'Successful original32-update canonical40/public receipts, passed selected-trajectory audit and matching source SHAs.',
        'Preserve old025 attack-first selection and older six/seven-source declarations; recompute all measured original128 assessments and rank complete, recovery, air, attack, declared order. No failed source is discarded.',
        'Only after ordinary method selection, integrate reviewed helper/trainer/config in original main, permit only those two declared runtime changes against all284 comparison inputs, and rerun relevant CPU checks.',
        'Original shared main GPU0 guard/ledger only; actual8-update pilot then full128 and actual six-case video. No candidate-local native job, automatic long extension or upload.']
    new['planned_native_pilot']['output'] = str(reports/'residual_ppo_centered_leg06_after075_intermediates_n2048_probe8_01')
    new['subsequent_assessment']['output'] = str(reports/'residual_ppo_centered_leg06_after075_intermediates_n2048_probe8_fixed_train128_01')
    dependencies = [Path(__file__), old_path, workflow/'run.py', workflow/'check.py',
        workflow/'ACTUAL_CPU_PIPELINE_CHECK.json', comparison/'PREDECLARED_PLAN.json',
        owner/'PREDECLARED_PLAN.json', analysis, reports/'placement075_action_authority_20261026/ACTUAL_PROGRESS_CUSTODY_CHECK.json',
        ROOT/'configs/g07_full_racket_residual_ppo_centered_leg06_placement075_v4.json',
        ROOT/'scripts/initialize_centered_action_units_v4.py', ROOT/'scripts/train_full_racket_residual_ppo_v4.py',
        ROOT/'tests/test_initialize_centered_action_units_v4.py']
    new['source_sha256'] = {str(p): sha(p) for p in dependencies}
    with output.open('x') as stream:
        json.dump(new, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status=new['status'], known_sources=6, pending_sources=4,
        live_owner_PIDs=[1734658, 1744903], main_held_inputs_unchanged=len(held), new_native_execution=False)))


if __name__ == '__main__':
    main()
