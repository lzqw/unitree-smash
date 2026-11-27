"""Paired full-tail fixed128 and strict independent512 evaluation for three frozen head transfers."""
import argparse
import hashlib
import json
from pathlib import Path

from quadsmash.rl.supermimic_independent_panel import load_panel
from quadsmash.rl.supermimic_terminal_scoring import STRICT_SCORING
from scripts.compare_supermimic_fixed_stages import load_run
from scripts.evaluate_supermimic_independent512 import collect_scores
from scripts.merge_approach_recovery_heads import merge
from scripts.run_recovery_training_round import run_stage
from scripts.run_supermimic_recovery_baseline import dump, summarize


def paired_changes(before, after):
    old_rows, reference, _ = load_run(before)
    new_rows, candidate, _ = load_run(after)
    if reference != candidate or any(r['numerical_failures'] for r in old_rows + new_rows):
        raise ValueError('Paired transfer evaluations differ in native conditions or are invalid')
    if json.loads((before/'actuator_contract.json').read_text()) != json.loads((after/'actuator_contract.json').read_text()):
        raise ValueError('Native drive parameters changed')
    new_by_id = {r['case_id']: r for r in new_rows}
    result = {}
    for field in ('qualified_airborne_string_contact', 'ball_side_complete_except_recovery', 'complete5'):
        result[field] = {name: [r['case_id'] for r in old_rows if bool(r[field]) == previous
                               and bool(new_by_id[r['case_id']][field]) != previous]
                         for name, previous in [('gained', False), ('lost', True)]}
    # The strict scorer keeps robot faults in the full native event records.
    events = [json.loads(line) for line in (after/'episodes.jsonl').read_text().splitlines()]
    if len(events) != 128 or {r['case_id'] for r in events} != set(new_by_id):
        raise ValueError('Incomplete first-episode native fault records')
    result['robot_net'] = [r['case_id'] for r in events if r['robot_net_fault']]
    result['no_ball_regression'] = (not result['qualified_airborne_string_contact']['lost']
                                    and not result['ball_side_complete_except_recovery']['lost']
                                    and not result['robot_net'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('recovery', 'recovery-fixed128', 'final-best', 'h8d', 'champion', 'output', 'gpu-lock'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    source = json.loads((args.recovery_fixed128/'metrics.json').read_text())
    scores = summarize(args.recovery_fixed128)
    if (source['evaluated_checkpoint_sha256'] != hashlib.sha256(args.recovery.read_bytes()).hexdigest()
            or scores['qualified_contacts'] != 108 or scores['body_ground_given_contact'] > 30
            or scores['strict_physical_recovery'] < 50 or scores['relaxed_physical_recovery'] < 65
            or scores['relaxed_ball_side'] < 95 or scores['robot_net']):
        raise ValueError('The selected Recovery checkpoint has not reached the scale-training milestone')
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    status = root/'status.json'
    panel = load_panel()
    runs = {}
    with args.gpu_lock.open('a') as lock:
        for label, approach in [('final_best', args.final_best), ('h8d', args.h8d), ('champion', args.champion)]:
            work = root/label
            work.mkdir()
            merged = work/'merged.pt'
            merge(approach, args.recovery, merged)
            runs[label] = (approach, merged)
            results = []
            for name, checkpoint in zip(('original','merged'), runs[label], strict=True):
                results.append(run_stage(['scripts/evaluate_supermimic_v1.py','--checkpoint',str(checkpoint),
                    '--headless','--kit_args=--renderer/multiGpu/enabled=false','--device','cuda:0','--physics-device','cuda:0',
                    '--seed','882072','--research-fixed128','--recovery-full-tail'], work/('fixed128_'+name), lock, status))
            changes = paired_changes(*results)
            dump(work/'fixed128.json', dict(original=summarize(results[0]), merged=summarize(results[1]), changes=changes))
            if not changes['no_ball_regression']:
                dump(status,dict(state='fixed128_transfer_failed', approach=label))
                return
        # Fix all policies on the development panel before opening independent cases.
        for label, policies in runs.items():
            work = root/label
            batches = []
            for name, checkpoint in zip(('original','merged'), policies, strict=True):
                folder = work/('independent_'+name)
                folder.mkdir()
                batch_paths = []
                for i in range(4):
                    path = run_stage(['scripts/evaluate_supermimic_v1.py','--checkpoint',str(checkpoint),
                        '--headless','--kit_args=--renderer/multiGpu/enabled=false','--device','cuda:0','--physics-device','cuda:0',
                        '--seed',str(panel['native_reset_seed']),'--official-full-episode','--independent-panel-batch',str(i)],
                        folder/f'native_{i}',lock,status)
                    (folder/f'batch_{i}').symlink_to(path.name)
                    batch_paths.append(path)
                metrics = collect_scores(folder, hashlib.sha256(checkpoint.read_bytes()).hexdigest(), panel)
                if metrics['scoring_schema'] != STRICT_SCORING or metrics['numerical_failures']:
                    raise ValueError('Expected official strict, numerically valid independent512 results')
                dump(folder/'independent512.json',metrics)
                batches.append(batch_paths)
            changes = [paired_changes(a,b) for a,b in zip(*batches,strict=True)]
            dump(work/'paired_independent512.json',dict(batches=changes,passed=all(r['no_ball_regression'] for r in changes)))
        passed = all(json.loads((root/k/'paired_independent512.json').read_text())['passed'] for k in runs)
        dump(status,dict(state='transfer_evaluated',passed=passed))


if __name__ == '__main__':
    main()
