"""Register a conditional TRAIN reward revision using completed native evidence."""
from datetime import datetime, timezone
import copy
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
MAIN = ROOT.parent/'unitree_smash'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from quadsmash.control.measured_joint_placement_reward_v4 import MeasuredJointPlacementRewardV4, placement_settings


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = HERE/'PREDECLARED_PLAN.json'
    assert not output.exists()
    config_path = ROOT/'configs/g07_full_racket_residual_ppo_centered_placement075_v4.json'
    original_path = ROOT/'configs/g07_full_racket_residual_ppo_centered_v4.json'
    original, config = read(original_path), read(config_path)
    expected = copy.deepcopy(original)
    expected['joint_placement_reward']['distance_scale_m'] = .75
    expected['runner']['seed'] = 748901
    assert config == expected and placement_settings(config['joint_placement_reward']) == {'weight': 20., 'distance_scale_m': .75}
    reports = MAIN/'reports/G07_full_racket_v4'
    evidence_path = reports/'centered8_to40_fixed_diagnostic_20261026/REPORT.json'
    evidence = read(evidence_path)
    attacks = evidence['actual_centered40_attack_cases']
    assert [case['TRAIN'] for case in attacks] == [1, 12, 113, 121]
    assert all(case['crossed_net'] and not case['landed_target'] and not case['independent_outcome']['fault'] for case in attacks)
    landing = torch.tensor([case['first_landing_m'] for case in attacks], dtype=torch.float32)
    court_path = MAIN/'configs/court_net_v4.json'
    court = read(court_path)
    ones = torch.ones(len(attacks), dtype=torch.bool)
    zeros = torch.zeros_like(ones)
    replay = []
    for settings in [original['joint_placement_reward'], config['joint_placement_reward']]:
        reward = MeasuredJointPlacementRewardV4(len(attacks), 'cpu', court, settings)
        replay.append(reward.step(ground_seen=ones, legal_air=ones, attack=ones, crossed=ones,
                                  faulted=zeros, first_landing=landing).tolist())
    feedback = [dict(TRAIN=case['TRAIN'], actual_first_landing_m=case['first_landing_m'],
                     original025_formula_reward=replay[0][index], proposed075_formula_reward=replay[1][index],
                     original_complete_success=False) for index, case in enumerate(attacks)]
    assert all(0. <= item['original025_formula_reward'] <= item['proposed075_formula_reward'] < 20. for item in feedback)
    cpu_path = HERE/'ACTUAL_FROZEN40_CPU_COMPATIBILITY.json'
    cpu = read(cpu_path)
    assert cpu['source_public_mean_bitwise_preserved'] and cpu['actual_public_inputs'] == 1890
    assert cpu['source_actor_normalizer_count_before_new_learning'] == 10485760
    assert not cpu['new_native_execution']
    selection_plan_path = reports/'centered_intermediate_selection_queue01/PREDECLARED_PLAN.json'
    selection_plan = read(selection_plan_path)
    process = Path('/proc/1697096/cmdline')
    assert process.exists() and b'centered_intermediate_selection_queue01/run.py' in process.read_bytes()
    held = selection_plan['source_hashes']
    for name, digest in held.items():
        path = Path(name)
        assert sha(path if path.is_absolute() else MAIN/path) == digest, name
    run = reports/'residual_ppo_centered_placement075_n2048_probe8_01'
    evaluation = reports/'residual_ppo_centered_placement075_n2048_probe8_fixed_train128_01'
    assert not run.exists() and not evaluation.exists()
    python = '/home/autodrive/Isaac/env_isaaclab/bin/python'
    train = [python, str(MAIN/'scripts/train_full_racket_residual_ppo_v4.py'), '--output', str(run),
             '--base-checkpoint', str(reports/'residual_ppo_joint_placement_actor32_n2048_probe8_01/policy.pt'),
             '--config', str(MAIN/config_path.relative_to(ROOT)), '--num-envs', '2048', '--updates', '8',
             '--initialize-centered-policy', 'SELECTED_MEASURED_PUBLIC_POLICY_FROM_RESULT',
             '--centered-source-evaluation', 'MATCHING_COMPLETED_ORIGINAL_TRAIN128_EVALUATION',
             '--seed', '748901', '--headless']
    guard_prefix = [python, str(MAIN/'scripts/run_gpu_job.py'), '--goal', 'G07', '--budget-window',
                    'g07-full-racket-unlimited-20261024']
    protocol_path = MAIN/'configs/g07_full_racket_eval_v4_r3.json'
    protocol = read(protocol_path)
    result = dict(schema='conditional-centered-placement075-train-revision-v1', created_at=datetime.now(timezone.utc).isoformat(),
                  status='PREPARED_CPU_CHECKED_NATIVE_NOT_RUN',
                  decision_timing='After completed frozen8/40 evidence, before pending16/24/32 outcomes. Development revision, not a blind test.',
                  hypothesis='Broaden bounded first-landing feedback after actual qualifying airborne attacks; the measured attacks overshoot the original target. This may help learning but does not itself solve air/recovery failures.',
                  configuration=str(config_path.relative_to(ROOT)), configuration_sha256=sha(config_path),
                  changed_configuration_fields={'joint_placement_reward.distance_scale_m': [.25, .75], 'runner.seed': [original['runner']['seed'], 748901]},
                  unchanged_training_settings='Same20 maximum placement weight, actual-attack/cross/first-ground eligibility,18 joint scales, architecture, PPO settings and two critic warmups.',
                  counterfactual_terminal_formula_replay=feedback,
                  replay_limit='Formula evaluations on real terminal event records, not observed historical reward payments, new trajectories or improved performance.',
                  source_selection='Use the actual completed original selection RESULT.json winner by its already registered rank; do not select40 merely because it is the CPU compatibility witness.',
                  launch_conditions=['Actual1697096 source owner exit; completed selection RESULT.json and terminal state agree; no active original GPU ledger job.',
                                     'If selected original TRAIN result meets50/70/85/NF0, skip this revision and proceed to original formal review.',
                                     'Otherwise require CONTINUE_METHOD_FROM_BEST_MEASURED_FROZEN, matching selected policy SHA and complete native original128 zero-noise evaluation.',
                                     'Root integrates the prepared initializer/trainer/test and this config into main only after owners exit; preserve the196 held sources until then.',
                                     'Replace both structured argument markers from verified actual results. Never execute unresolved markers; never treat a diagnostic checkpoint as canonical.'],
                  initialization='Transfer only the measured outer mean actor and its normalizer on the exact same old8 fixed base. Fresh critic/critic normalizer, exploration, Adam, RNG and learning counters; actual fresh ground states.',
                  normalizer_accounting='Actor starts at measured source countM and advances toM+new steps; critic/new learner start0. New iteration8 is this revision iteration8, not source iteration+8.',
                  first_native_checks=['2048 actual fresh ground states, all clocks/episode IDs start0, original128 cases repeated16 times, fixed base unchanged.',
                                       'First update262144 steps: copied actor remains through configured critic warmup, all tensors finite, actor normalizationM+262144 and critic262144.',
                                       'Original main GPU0 guard/ledger, enabled full trajectory recorder with verified public inputs, NF0; preserve any real failed attempt.'],
                  train_command_template=train,
                  guarded_train_template=guard_prefix+['--max-seconds','7200','--env-steps-cap','2097152','--output',str(run),'--require-artifact','completed.json','--']+train,
                  fixed_evaluation_command=[python, str(MAIN/'scripts/evaluate_full_racket_train_v4.py'), '--output', str(evaluation),
                                            '--checkpoint', str(run/'policy.pt'), '--noise', 'none', '--seed', '724201', '--headless'],
                  fixed_evaluation_guard=dict(max_seconds=2700, env_steps_cap=102400, original_main_shared_window=True),
                  post_training='Verify canonical new8, fresh-stage episode chains, source-offset normalizers, all selected native traces and independent changing-policy counts; then frozen original128 and the same six actual side/top cases with failures.',
                  next_decision='Use actual pilot8 frozen evidence before any longer continuation. If ready, lock one source and evaluate original A/B200 each; otherwise continue method work under the full goal.',
                  original_gates=dict(worse_panel_complete_rate=.50, worse_panel_air_rate=.70, worse_panel_conditional_recovery_rate=.85, numerical_failures=0, panel_episodes=200, panels=2),
                  resource=dict(gpu=0, cumulative_GPU_hour_limit=None, calendar_deadline=None, original_shared_guard_and_ledger_only=True, cloud_resources=False),
                  protocol_sha256=sha(protocol_path), asset_sha256=protocol['asset']['sha256'], main_held_sources_verified=len(held),
                  CPU_witness_is_not_selected_source=True, CPU_mean_replay_inputs=1890,
                  source_sha256={str(p):sha(p) for p in [Path(__file__), config_path, original_path, evidence_path, cpu_path, selection_plan_path, court_path, protocol_path,
                                                       ROOT/'scripts/initialize_centered_policy_v4.py', ROOT/'scripts/train_full_racket_residual_ppo_v4.py']},
                  new_native_execution=False, new_training=False, formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(configuration_sha256=result['configuration_sha256'], feedback=feedback, main_held_sources_verified=len(held),
                         launch_status=result['status'], new_native_execution=False), indent=2))


if __name__ == '__main__':
    main()
