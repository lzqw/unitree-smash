"""Create a separate pinned audit entry without changing the running producer."""
from datetime import datetime, timezone
import ast
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / 'scripts/train_full_racket_origin0_residual_ppo_v4.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (HERE / 'ENTRY_PLAN.json').exists()
    admission = json.loads((ROOT / 'reports/G07_full_racket_v4/origin0_residual_after_gain_body_queue01/ADMISSION_DECISION.json').read_text())
    assert sha(SOURCE) == admission['source_sha256'][str(SOURCE)]
    source = SOURCE.read_text()
    original_root = "ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))"
    audit_root = "ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))"
    before = '        initial_update=runner.current_learning_iteration;history=[];interrupted=False'
    hook = '''        from reports.G07_full_racket_v4.recovery_sequence_observer_20261027.recorder import RecoverySequenceRecorder
        import zipfile
        audit_here=Path(__file__).resolve().parent
        audit_plan=json.loads((audit_here/'ENTRY_PLAN.json').read_text())
        for name,expected in audit_plan['runtime_source_sha256'].items():
            if digest(Path(name))!=expected:raise ValueError('Read-only audit source changed: '+name)
        if args.num_envs!=128 or args.updates!=7 or args.seed!=751001 or args.resume is not None:
            raise ValueError('Require the original fresh128/7warmup/seed751001 replay, not another course')
        if digest(args.base_checkpoint)!='204ff76f21625d49cd1dc29ec3b6256447a5fa8b2fec935dda986842fc330034':
            raise ValueError('Require the actual original unchanged source')
        observer_sources=[Path(name) for name in audit_plan['runtime_source_sha256']]
        observer_sources.append(audit_here/'ENTRY_PLAN.json')
        with zipfile.ZipFile(args.output/'read_only_observer_source.zip','x',zipfile.ZIP_DEFLATED)as archive:
            for path in observer_sources:archive.write(path,str(path.relative_to(ROOT)))
        atomic_json(args.output/'READ_ONLY_OBSERVER_INPUTS.json',dict(
            entry_plan_sha256=digest(audit_here/'ENTRY_PLAN.json'),runtime_source_sha256=audit_plan['runtime_source_sha256'],
            selected_first_episode_lanes=[69,112],training_or_scoring_changed=False,
            source_first128_reference_sha256=audit_plan['original_first128_sha256'],
            physics_step_requested_by_observer=False,formal_acceptance=False))
        sequence_observer=RecoverySequenceRecorder(env,args.output/'recovery_sequence',
            lanes=(69,112),maximum_samples=147456,native_source_confirmed=True)
        env._diagnostic_observer=sequence_observer
'''
    finish_before = '        if interrupted:\n'
    finish_hook = '        sequence_observer.finish(require_complete=not interrupted)\n'
    assert source.count(original_root) == source.count(before) == source.count(finish_before) == 1
    revised = source.replace(original_root, audit_root, 1).replace(before, hook + before, 1).replace(finish_before, finish_hook + finish_before, 1)
    ast.parse(revised)
    reversed_copy = revised.replace(audit_root, original_root, 1).replace(hook, '', 1).replace(finish_hook, '', 1)
    assert reversed_copy == source
    output = HERE / 'audit_training_entry.py'
    with output.open('x') as stream:
        stream.write(revised)
    with (HERE / 'AUDIT_ENTRY_DIFF.patch').open('x') as stream:
        stream.write(''.join(difflib.unified_diff(source.splitlines(True), revised.splitlines(True),
            fromfile='original_actual_training_entry.py', tofile='audit_training_entry.py')))
    reference = ROOT / 'reports/G07_full_racket_v4/origin0_residual_warmup7_actor8_20261027/FIRST128_EPISODES.json'
    paths = [SOURCE, output, HERE / 'recorder.py', HERE / 'CPU_INTERFACE_CHECK.json']
    plan = dict(prepared_at=datetime.now(timezone.utc).isoformat(), runtime_source_sha256={str(p): sha(p) for p in paths},
        original_first128_reference=str(reference), original_first128_sha256=sha(reference),
        original_entry_recovered_exactly_after_removing_only_observer_hooks_and_restoring_ROOT=True,
        audit_difference='Report-directory ROOT, source checks and source snapshot, observer attachment and finish only.',
        source_or_scoring_files_overwritten=False, new_native_execution=False,
        requires_actual_current_owner_exit=True,
        required_terminal_state='COMPLETED_ORIGIN0_RESIDUAL15_AND_COLD128',
        future_replay=dict(num_envs=128,updates=7,seed=751001,env_steps_cap=114688,max_seconds=5400),
        no_same_trajectory_or_causal_recovery_claim_before_actual_comparison=True,
        formal_cases_read=False, formal_acceptance=False)
    with (HERE / 'ENTRY_PLAN.json').open('x') as stream:
        stream.write(json.dumps(plan, indent=2) + '\n')
    assert sha(SOURCE) == admission['source_sha256'][str(SOURCE)]
    print(json.dumps(dict(prepared=str(output), original_producer_unchanged=True,
        exact_reversible_hook_edit=True, new_native_execution=False)), flush=True)


if __name__ == '__main__':
    main()
