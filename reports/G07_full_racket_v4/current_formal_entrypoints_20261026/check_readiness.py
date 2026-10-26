"""Check the formal admission path against an actually unqualified policy.

This must fail closed before opening any formal case or writing a policy lock.
The formal evaluator is not executed and no simulated result is manufactured.
"""
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
MAIN = ROOT.parent/'unitree_smash'
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(MAIN))
import quadsmash.control


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    assert not (HERE/'PLAN.json').exists()
    aim_path = ROOT/'quadsmash/control/full_racket_contact_aim_v4.py'
    lock_path = ROOT/'scripts/lock_full_racket_current_policy_v4.py'
    evaluate_path = ROOT/'scripts/evaluate_full_racket_current_policy_v4.py'
    load('quadsmash.control.full_racket_contact_aim_v4',aim_path)
    lock = load('current_policy_formal_lock_candidate',lock_path)
    lock.ROOT = MAIN
    protocol_path = MAIN/'configs/g07_full_racket_eval_v4_r3.json'
    protocol = read(protocol_path)
    selected = read(MAIN/'reports/G07_full_racket_v4/body_residual48_after_grid16_queue01/ADMISSION_DECISION.json')['source']
    evaluation,policy = Path(selected['evaluation']),Path(selected['policy'])
    forbidden = {(MAIN/suite['path']).resolve() for suite in protocol['formal_suites'].values()}
    attempts = []

    def audit(event,args):
        if event == 'open' and isinstance(args[0],(str,bytes)):
            candidate = Path(args[0]).resolve()
            if candidate in forbidden:
                attempts.append(str(candidate))
                raise AssertionError('Formal contents must remain unopened before readiness')

    sys.addaudithook(audit)
    sources = [Path(__file__),aim_path,lock_path,evaluate_path,protocol_path,policy,
        *[evaluation/name for name in ['INPUTS.json','metrics.json','episodes.json','job.json','completed.json']]]
    plan = dict(created_at=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(path):sha(path) for path in sources},
        actual_source_original_summary=selected['summary'],
        expected='Reject actual below-threshold TRAIN128 source before all formal case access or lock output.',
        new_native_execution=False,formal_cases_read=False,formal_acceptance=False)
    (HERE/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    output = HERE/'REJECTED_FORMAL_LOCK'
    assert not output.exists()
    sys.argv = [str(lock_path),'--checkpoint',str(policy),'--training-evaluation',str(evaluation),
        '--protocol',str(protocol_path),'--output',str(output)]
    try:
        lock.main()
    except ValueError as error:
        reason = str(error)
        assert 'does not meet50/70/85 and NF0 readiness' in reason,reason
    else:
        raise AssertionError('Actual unqualified source unexpectedly admitted')
    assert not attempts and not output.exists()
    assert all(sha(Path(name)) == digest for name,digest in plan['source_sha256'].items())
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(),
        actual_original_TRAIN128_source_rejected=True,reason=reason,
        original_source_complete=selected['summary']['complete'],
        original_source_air=selected['summary']['airborne_contacts'],
        original_source_conditional_recovery=selected['summary']['conditional_recovery_count'],
        audited_formal_file_open_attempts=attempts,policy_lock_created=False,
        current_formal_evaluator_native_execution=False,
        scope='Actual real-source readiness rejection and formal-file access guard only; positive admission/native A/B still unrun.',
        new_native_execution=False,formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    (HERE/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
