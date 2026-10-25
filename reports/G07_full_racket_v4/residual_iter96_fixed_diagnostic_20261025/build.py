"""Recompute both complete TRAIN panels before choosing the next initialization."""
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = ROOT / 'reports/G07_full_racket_v4'
sys.path.insert(0, str(ROOT))
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases, summarize_training
from quadsmash.evaluation_v4 import independent_episode


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def main():
    output = HERE / 'REPORT.json'
    table = HERE / 'PAIRED_TRAIN128_OUTCOMES.csv'
    assert not output.exists() and not table.exists(), 'Never overwrite a diagnostic'
    protocol_path = ROOT / 'configs/g07_full_racket_eval_v4_r3.json'
    protocol = read(protocol_path)
    protocol['_root'] = str(ROOT)
    cases, indices = training_cases(ROOT, protocol)
    assert indices == list(range(128))
    results, outcomes, files = {}, {}, {}
    for label, name in [('iter32', 'residual_ppo_fixed_train128_iter32_n512_01'),
                        ('iter96', 'residual_ppo_fixed_train128_iter96_n2048_01')]:
        run = BASE / name
        job, inputs, rows = map(read, [run / 'job.json', run / 'INPUTS.json', run / 'episodes.json'])
        assert job['status'] == 'SUCCEEDED' and job['charged_env_steps'] == 102400
        assert inputs['training_indices'] == indices and inputs['protocol_sha256'] == sha(protocol_path)
        assert all(float(value) == 0 for value in inputs['reset_noise_amplitudes'].values())
        policy = Path(job['command'][job['command'].index('--checkpoint') + 1])
        if not policy.is_absolute():
            policy = ROOT / policy
        assert inputs['policy_sha256'] == sha(policy)
        summary = summarize_training(rows, cases, indices, protocol)
        summary.pop('distinct_TRAIN_indices')
        checked = [independent_episode(row, protocol) for row in rows]
        detail = []
        for row, check in zip(rows, checked):
            velocity = row['outgoing_velocity_mps']
            attack = bool(check['airborne_contact'] and row['released']
                and sum(value * value for value in velocity) ** .5 >= 8
                and velocity[0] > 0 and velocity[2] <= -1)
            detail.append(dict(air=check['airborne_contact'], recovery=check['conditional_recovery'],
                complete=check['complete'], attack=attack, target=row['landed_target']))
        results[label] = dict(summary=summary, policy_sha256=inputs['policy_sha256'],
            actual_gpu_hours=job['charged_gpu_hours'], reset_noise_seed=inputs['reset_noise_seed'],
            attack_target_overlap=sum(item['attack'] and item['target'] for item in detail),
            attack_and_recovery=sum(item['attack'] and item['recovery'] for item in detail))
        outcomes[label] = detail
        for path in [run / 'job.json', run / 'INPUTS.json', run / 'episodes.json', run / 'initial_states.json']:
            files[str(path.relative_to(ROOT))] = sha(path)
    assert results['iter32']['summary']['complete'] == results['iter96']['summary']['complete'] == 0
    parent = read(BASE / 'residual_resize2048_queue01/state.json')
    current = dict(parent['independent_fixed_TRAIN128'])
    current.pop('distinct_TRAIN_indices')
    assert current == results['iter96']['summary']
    formal = read(BASE / 'residual_iter96_formal_queue01/state.json')
    assert formal['state'] == 'SKIPPED_TRAIN_BELOW_ORIGINAL_THRESHOLDS' and not formal['formal_cases_read']
    paired = []
    for index in indices:
        item = dict(training_index=index, case_id=cases[index]['case_id'])
        for label in outcomes:
            item.update({label + '_' + key: value for key, value in outcomes[label][index].items()})
        paired.append(item)
    with table.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(paired)
    actor_source = BASE / 'residual_ppo_n512_resume24_01/checkpoint.pt'
    report = dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='Two fixed policies on all original128 TRAIN cases, zero reset noise. Descriptive training diagnostics, not formal A/B or a causal method comparison.',
        results=results, source_files=files, paired_outcomes_file=table.name,
        paired_outcomes_sha256=sha(table), next_actor_initialization=dict(iteration=32,
            canonical_checkpoint=str(actor_source.relative_to(ROOT)), sha256=sha(actor_source),
            reason='Both policies have zero complete outcomes. Actor32 has14 qualified recoveries and3 attacking returns, versus8 and1 for actor96. Actor32 has fewer air contacts22 versus24. Choose actor32 for the next fresh joint-placement reward experiment based on this TRAIN trade-off; no statistical superiority claim.',
            fresh_critic_optimizer_and_counters=True, original_physics_rules_cases_and_acceptance_unchanged=True),
        formal_queue_state=formal['state'], formal_cases_read=False,
        formal_acceptance=False, goal_complete=False)
    temporary = output.with_suffix('.json.tmp')
    assert not temporary.exists()
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(output)
    print(json.dumps(dict(results=results, next_actor_source=report['next_actor_initialization'])))


if __name__ == '__main__':
    main()
