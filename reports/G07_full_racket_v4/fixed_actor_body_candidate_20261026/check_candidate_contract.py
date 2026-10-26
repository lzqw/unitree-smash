"""Check declared proposals, real source scores and unchanged cold evaluation."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import py_compile
import sys
import torch

ROOT = Path(__file__).resolve().parents[3]
MAIN = ROOT.parent/'unitree_smash'
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from quadsmash.control.full_racket_body_search_v4 import (
    COORDINATES, INITIAL_SIGMA, initial_population, next_population, summarize_candidate)
from quadsmash.control.full_racket_train_evaluation_v4 import summarize_training, training_cases


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result_path = HERE/'CANDIDATE_CONTRACT_CHECK.json'; assert not result_path.exists()
    source = MAIN/'reports/G07_full_racket_v4/residual_ppo_centered_placement075_n2048_probe8_01/policy.pt'
    checkpoint = torch.load(source,map_location='cpu',weights_only=False)
    identity = read(HERE/'ACTUAL_SOURCE_IDENTITY_CHECK.json')
    assert identity['actual_source_policy_sha256'] == sha(source)
    assert identity['full800step_source_identity_max_action_error'] == 0.
    assert identity['source_sha256'][str(ROOT/'quadsmash/control/full_racket_fixed_actor_body_v4.py')] == sha(ROOT/'quadsmash/control/full_racket_fixed_actor_body_v4.py')
    reference = checkpoint['parameters']; population = initial_population(reference)
    assert torch.equal(population[0],reference) and torch.equal(population[1],reference)
    handover = population[:,2]+population[:,3]/population[:,4]
    assert torch.allclose(handover, handover[0].expand(10),atol=4e-7,rtol=0)
    protocol = read(MAIN/'configs/g07_full_racket_eval_v4_r3.json');protocol['_root'] = str(MAIN)
    cases, indices = training_cases(MAIN,protocol)
    runs = ['residual_ppo_centered_placement075_n2048_probe8_fixed_train128_01',
        'residual_ppo_centered_leg06_after075_intermediates_n2048_probe8_fixed_train128_01',
        'residual_ppo_centered_leg06_full_cohort_warmup7_n2048_probe13_fixed_train128_01']
    scores = []
    for run in runs:
        directory = MAIN/'reports/G07_full_racket_v4'/run
        rows = read(directory/'episodes.json')
        actual = summarize_training(rows,cases,indices,protocol)
        scored = summarize_candidate([dict(row,closest_string_gap_m=0.) for row in rows],protocol)
        for key in ['complete','airborne_contacts','conditional_recovery_count','attack_outgoing','numerical_failures']:
            assert scored[key] == actual[key]
        scores.append(dict(run=run,actual=actual,CPU_rank_example_with_assumed_zero_gap=scored))
    assert [(x['actual']['complete'],x['actual']['airborne_contacts'],x['actual']['conditional_recovery_count'],x['actual']['attack_outgoing']) for x in scores] == [(0,26,19,5),(0,27,14,6),(0,25,14,1)]
    gen = torch.Generator().manual_seed(750101)
    second, mean, sigma = next_population(reference,population,[5,7,0,1,2,3,4,6,8,9],
        reference[list(COORDINATES)],torch.tensor(INITIAL_SIGMA),gen)
    assert torch.equal(second[0],reference) and torch.equal(second[1],population[5])
    assert torch.isfinite(second).all() and (sigma > 0).all()
    old = (ROOT/'scripts/evaluate_full_racket_train_v4.py').read_text()
    new = (ROOT/'scripts/evaluate_full_racket_fixed_actor_body_train_v4.py').read_text()
    assert new == old.replace('from quadsmash.control.full_racket_residual_v4 import make_rally_controller',
        'from quadsmash.control.full_racket_fixed_actor_body_v4 import make_rally_controller')
    paths = [Path(__file__),ROOT/'quadsmash/control/full_racket_fixed_actor_body_v4.py',
        ROOT/'quadsmash/control/full_racket_body_search_v4.py',ROOT/'scripts/search_full_racket_fixed_actor_body_v4.py',
        ROOT/'scripts/evaluate_full_racket_fixed_actor_body_train_v4.py']
    for path in paths: py_compile.compile(str(path),doraise=True)
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(path):sha(path) for path in paths},actual_source_policy_sha256=sha(source),
        original128_scores_independently_recomputed=scores,
        two_exact_reference_candidates=True,eight_declared_body_candidates=True,
        recovery_handover_max_error_s=float((handover-handover[0]).abs().max()),
        recovery_handover_tolerance_s=4e-7,
        second_generation_reference_elite_and_coordinate_mask_checked=True,
        cold_evaluator_only_change_is_explicit_controller_factory_import=True,
        source_identity_actual_public_rows=1600,source_identity_max_action_error=0.,
        gap_for_CPU_ranking_example_is_assumed_zero_not_measured=True,
        native_candidate_performance_unproven=True,new_native_execution=False,
        formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    result_path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(source_identity_error=0.,actual_source_air=26,actual_source_recovered=19,
        cold_evaluation_task_criteria_unchanged=True,proposal_count=10)),flush=True)


if __name__ == '__main__': main()
