"""Register a conditional larger-leg TRAIN candidate before075 frozen results."""
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
MAIN = ROOT.parent/'unitree_smash'


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = HERE/'CONDITIONAL_PLAN.json';assert not output.exists()
    reports = MAIN/'reports/G07_full_racket_v4'
    active = reports/'centered_placement075_pilot8_queue01'
    evaluation075 = reports/'residual_ppo_centered_placement075_n2048_probe8_fixed_train128_01'
    assert not (evaluation075/'metrics.json').exists(), 'Do not relabel preparation after seeing075 frozen results'
    for pid,needle in [(1722226,'centered_placement075_pilot8_queue01/run.py'),(1722244,'residual_ppo_centered_placement075_n2048_probe8_01')]:
        path = Path('/proc')/str(pid)/'cmdline';assert path.exists() and needle.encode() in path.read_bytes()
    held = read(active/'PREDECLARED_PLAN.json')['source_hashes']
    held[str((active/'PREDECLARED_PLAN.json').relative_to(MAIN))] = sha(active/'PREDECLARED_PLAN.json')
    for name,digest in held.items():
        path = Path(name) if Path(name).is_absolute() else MAIN/name
        assert sha(path) == digest,name
    config_path = ROOT/'configs/g07_full_racket_residual_ppo_centered_leg06_placement075_v4.json'
    old = read(ROOT/'configs/g07_full_racket_residual_ppo_centered_placement075_v4.json')
    new = read(config_path);expected = copy.deepcopy(old)
    expected['joint_scale_rad'] = [.06]*12+[.15]*6;expected['runner']['seed'] = 749301
    assert new == expected
    cpu_path = HERE/'ACTUAL_FROZEN40_CPU_ACTION_UNIT_CHECK.json';cpu = read(cpu_path)
    assert cpu['actual_total_witness_rows'] == 3490 and cpu['actual_physical_residual_means_bitwise_equal']
    assert cpu['actual_physical_residual_mean_max_error_rad'] == 0 and cpu['fresh_Adam_entries'] == 0
    early_path = reports/'centered_placement075_health_20261026/ACTUAL_UPDATE3_POST_WARMUP_CHECK.json'
    early = read(early_path);selection_path = reports/'centered_intermediate_selection_queue01/RESULT.json'
    selection = read(selection_path)
    original_candidates = [dict(candidate_id=f'centered{entry["iteration"]}',
        policy=entry['policy'],summary=entry['summary']) for entry in sorted(selection['all_candidates'],key=lambda v:v['iteration'])]
    protocol_path = MAIN/'configs/g07_full_racket_eval_v4_r3.json';protocol = read(protocol_path)
    sources = [Path(__file__),config_path,cpu_path,early_path,selection_path,protocol_path,
        active/'PREDECLARED_PLAN.json',ROOT/'scripts/initialize_centered_action_units_v4.py',
        ROOT/'scripts/train_full_racket_residual_ppo_v4.py',ROOT/'tests/test_initialize_centered_action_units_v4.py']
    result = dict(created_at=datetime.now(timezone.utc).isoformat(),status='CPU_PREPARED_NOT_MAIN_INTEGRATED_NOT_NATIVE_RUN',
        decision_timing='After known five-source failures and actual075 early training boundary3, before completed075 frozen results. Development method candidate, not blind evidence.',
        hypothesis='Offer more leg residual authority for coupled landing/recovery while initializing the measured physical mean on observed inputs. This does not by itself solve missed strings contacts or attack placement.',
        evidence_scope='Boundary3 closed episodes are biased toward early failures while756 original first episodes remain open; do not treat partial training counts as frozen rates.',
        configuration=str(config_path.relative_to(ROOT)),configuration_sha256=sha(config_path),
        changed_configuration_fields={'joint_scale_rad.legs':[.03,.06],'runner.seed':[748901,749301]},
        additional_initialization_change='Explicit final actor affine leg rows/bias times0.5; fresh leg std0.05,arm std0.1. Actor normalizer retained; fresh critic/critic normalization/Adam/RNG/counters and actual ground.',
        physical_mean_equivalence_scope='Measured input witnesses only. Source leg saturation on that witness is rejected before mutation; unseen source-clipped means can differ when bounds expand.',
        fresh_preclipping_physical_exploration_std_rad={'legs':.003,'arm':.015},
        source_selection_after_actual075_completion=dict(
            candidate_order_for_final_tie=['centered8','centered16','centered24','centered32','centered40','placement075_stage8'],
            ranking=['complete_count_desc','conditional_recovery_count_desc','airborne_contacts_desc','attack_outgoing_desc','declared_order_asc'],
            rationale='Recovery count and air coverage are prioritized for this recovery-authority candidate; this is a new declared development ranking, not a rewrite of the prior attack-priority selector.',
            known_original_candidates=original_candidates,
            pending_candidate=dict(candidate_id='placement075_stage8',policy=str(reports/'residual_ppo_centered_placement075_n2048_probe8_01/policy.pt'),evaluation=str(evaluation075)),
            required='Recompute all six full original128 zero-noise assessments and matching native receipts/policy SHA. If any candidate meets original50/70/85/NF0, skip this revision and preserve that frozen source for original formal review.'),
        CPU_witness40_is_not_selected_future_source=True,
        launch_requirements=['Actual1722226 observer exits after current native/fixed/video pipeline; no active original main ledger job. Hold all212 current sources until then.',
            'Root chooses from actual completed original128 results using this declared rank; do not borrow source evaluation or resume a diagnostic checkpoint.',
            'Integrate reviewed candidate code into original main, run relevant checks there and bind concrete selected weights/evaluation/input witness SHA and main sources/assets before native launch.',
            'Run only original main GPU0 guard/ledger with existing unlimited cumulative authorization; preserve all charges and any actual failed attempt.',
            'No native execution is authorized by a CPU proof alone: this candidate is an ordinary method option under the existing user authorization, chosen only after the pending actual evidence.'],
        planned_native_pilot=dict(num_envs=2048,updates=8,seed=749301,env_steps_cap=2097152,max_seconds=7200,
            output=str(reports/'residual_ppo_centered_leg06_placement075_n2048_probe8_01'),
            initialization_flag='--initialize-centered-action-units',paired_evaluation_flag='--centered-source-evaluation',
            first_update='Actual fresh2048 ground,source actor RMS M+262144,critic262144,mapped actor weights preserved through warmup,fresh std0.05legs/0.1arm,8critic Adam20,NF0,full recording.',
            final_boundary='Canonical new8,total2097152,fresh2048 episode chains,actor RMS M+steps/criticsteps,9actor/logstd Adam120 and8critic Adam160,original fixed base unchanged and all selected actual traces audited.'),
        subsequent_assessment=dict(original_TRAIN_cases=128,noise='none',seed=724201,env_steps_cap=102400,max_seconds=2700,
            output=str(reports/'residual_ppo_centered_leg06_placement075_n2048_probe8_fixed_train128_01'),actual_six_case_video=[19,23,107,0,8,58]),
        unchanged='Physical asset/court/rules/limits/physics62.5us/control10ms,target rectangle,maximum placement20/scale0.75,arm scales0.15,network/PPO/two critic warmups,full original acceptance.',
        original_acceptance=dict(panel_episodes=200,panels=2,worse_complete_rate=.5,worse_airborne_rate=.7,worse_conditional_recovery_rate=.85,numerical_failures=0),
        protocol_sha256=sha(protocol_path),asset_sha256=protocol['asset']['sha256'],main_held_sources_unchanged=len(held),
        source_sha256={str(p):sha(p) for p in sources},new_native_execution=False,new_training=False,
        formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    with output.open('x') as file:json.dump(result,file,indent=2);file.write('\n')
    print(json.dumps(dict(status=result['status'],configuration_sha256=result['configuration_sha256'],held_sources=len(held),new_native_execution=False)))


if __name__ == '__main__':main()
