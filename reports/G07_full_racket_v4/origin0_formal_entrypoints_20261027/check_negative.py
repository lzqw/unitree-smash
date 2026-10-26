"""Reject actual unqualified sources without opening any formal case content."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
MAIN = ROOT.parent/'unitree_smash'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases, summarize_training


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    assert not (HERE/'NEGATIVE_CHECK.json').exists()
    protocol = read(ROOT/'configs/g07_full_racket_eval_v4_r3_origin0.json')
    forbidden = {str((root/suite['path']).resolve()) for root in [ROOT, MAIN] for suite in protocol['formal_suites'].values()}
    attempts = []
    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            name = str(Path(args[0].decode() if isinstance(args[0], bytes) else args[0]).resolve())
            if name in forbidden:
                attempts.append(name)
                raise AssertionError('Formal case content must stay unopened for unqualified sources')
    sys.addaudithook(audit)
    spec = importlib.util.spec_from_file_location('origin0_formal_lock_negative', ROOT/'scripts/lock_full_racket_origin0_current_policy_v4.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    run = MAIN/'reports/G07_full_racket_v4/prepose_origin0_8_sequential_n128_01'
    checked = dict(protocol, _root=str(MAIN))
    cases, indices = training_cases(MAIN, checked)
    summary = summarize_training(read(run/'episodes_000.json'), cases, indices, checked)
    assert summary['complete'] == summary['airborne_contacts'] == 0 and summary['numerical_failures'] == 0
    try:
        module.require_original_thresholds(summary, protocol)
    except ValueError as error:
        original_threshold_rejection = str(error)
    else:
        raise AssertionError('Actual origin0 reference0/128 passed original readiness thresholds')
    old_policy = MAIN/'reports/G07_full_racket_v4/contact_aim7_sequential_n128_01/best_candidate.pt'
    old_cold = MAIN/'reports/G07_full_racket_v4/contact_aim7_best_fixed_train128_01'
    try:
        module.require_ready_training(MAIN, protocol, old_policy, old_cold)
    except ValueError as error:
        old_layout_rejection = str(error)
    else:
        raise AssertionError('Old16m policy/evaluation was accepted as new origin0 readiness')
    assert not attempts
    paths = [ROOT/'scripts'/name for name in ['lock_full_racket_origin0_current_policy_v4.py',
        'evaluate_full_racket_origin0_current_policy_v4.py', 'aggregate_full_racket_origin0_eval_v4.py']]
    paths += [Path(__file__), run/'episodes_000.json', old_policy, *[old_cold/name for name in
        ['INPUTS.json', 'metrics.json', 'job.json', 'completed.json']]]
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), actual_origin0_reference128_summary=summary,
        actual_origin0_threshold_rejection=original_threshold_rejection, actual_old_layout_rejection=old_layout_rejection,
        actual_formal_file_open_attempts=attempts, actual_formal_file_open_count=0,
        original_thresholds_unchanged=protocol['thresholds'], source_sha256={str(path):sha(path) for path in paths},
        positive_readiness_and_native400_evaluation_pending=True,
        scope='Real completed TRAIN data negative admission and formal-file access guard only. No fabricated passing episodes or new physical execution.',
        new_native_execution=False, formal_cases_read=False, formal_acceptance=False)
    (HERE/'NEGATIVE_CHECK.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
