"""Verify completed window search/cold and the existing recovery-owner handoff."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
MAIN = HERE.parents[2]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def process(pid, ticks):
    path = Path('/proc', str(pid))
    stat = (path/'stat').read_text().rsplit(')', 1)[1].split()
    assert stat[19] == ticks and stat[0] != 'Z'
    cmd = (path/'cmdline').read_bytes().replace(b'\0', b' ').decode()
    assert str(MAIN) in cmd
    env = (path/'environ').read_bytes().split(b'\0')
    cuda = next((v.split(b'=', 1)[1].decode() for v in env if v.startswith(b'CUDA_VISIBLE_DEVICES=')), None)
    children = [int(x) for x in (path/'task'/str(pid)/'children').read_text().split()]
    return dict(pid=pid, start_ticks=ticks, command=cmd, CUDA_VISIBLE_DEVICES=cuda,
                children=children, kernel_CPU_ticks=int(stat[11])+int(stat[12]))


def main():
    files = [Path(__file__)]
    for name in ['ACTUAL_COMPLETED_SEARCH_CHECK.json', 'ACTUAL_POLICY_REPLAY_AUDIT.json',
                 'ACTUAL_INDEPENDENT128_CHECK.json', 'ACTUAL_FAILURE_COMPONENT_CHECK.json',
                 'INDEPENDENT128_FAILURE_COMPONENT_CHECK.json']:
        path = HERE/name
        record = read(path)
        assert all(sha(Path(p)) == h for p, h in record['source_sha256'].items())
        files.append(path)
    cold = read(HERE/'ACTUAL_INDEPENDENT128_CHECK.json')
    summary = cold['independent_actual128_summary']
    assert summary['complete'] == 0 and summary['airborne_contacts'] == 14 and summary['conditional_recovery_count'] == 0
    assert summary['numerical_failures'] == 0
    window_state_path = BASE/'windowed_swing_queue01/state.json'
    assert read(window_state_path)['state'] == 'COMPLETED_WINDOWED_SWING7_AND_INDEPENDENT_COLD128'
    for pid in [1966524, 1994961, 1994976, 2011304, 2011319]:
        assert not Path('/proc', str(pid)).exists()
    owner = process(1968858, '70396314')
    assert owner['CUDA_VISIBLE_DEVICES'] == '' and 'recovery_prior_queue01/run.py' in owner['command']
    waiting = []
    for pid, ticks in [(1973878, '70601904'), (1984009, '71053999'),
                       (1990613, '71376602'), (2008628, '72257205')]:
        record = process(pid, ticks)
        assert record['CUDA_VISIBLE_DEVICES'] == '' and not record['children']
        waiting.append(record)
    recovery = BASE/'recovery_prior_queue01'
    state = read(recovery/'state.json')
    decision_path = recovery/'ACTUAL_COMPLETED_SOURCE_DECISION.json'
    decision = read(decision_path)
    assert decision['source_selected']
    assert decision['source_check']['source_policy_sha256'] == cold['selected_policy_sha256']
    assert decision['source_check']['source_cold_summary'] == summary
    files.append(decision_path)
    held = dict(read(BASE/'shared_target_bias_queue01/PLAN.json')['source_sha256'])
    receipts = [BASE/'shared_target_bias_queue01/PLAN.json']
    dynamic = recovery/'BOUND_SELECTED_SOURCE_INPUTS.json'
    if dynamic.exists():
        extra = read(dynamic)
        for p, h in extra.items():
            assert p not in held or held[p] == h
            held[p] = h
        receipts.append(dynamic)
    assert all(sha(Path(p)) == h for p, h in held.items())
    assert str(MAIN/'README.md') not in held and str(MAIN/'status/STATUS.json') not in held
    cpu_path = recovery/'SELECTED_SOURCE_CPU_CHECK.json'
    cpu_brief = None
    if cpu_path.exists():
        cpu = read(cpu_path)
        results = cpu['actual_selected_source_all9_public_replays']
        assert len(results) == 9 and results[0]['CPU_vs_selected_source_GPU_max_action_error'] <= .0005
        assert all(r['maximum_arm_target_rate_rad_s'] <= 5.0005 for r in results)
        cpu_brief = dict(actual_candidates_checked=9,
            actual_baseline_GPU_max_error=results[0]['CPU_vs_selected_source_GPU_max_action_error'])
        files.append(cpu_path)
    active = [r for r in read(MAIN/'status/GPU_LEDGER.json')['runs'] if r['status'] == 'RUNNING']
    assert len(active) <= 1
    native = []
    for row in active:
        assert row['goal'] == 'G07' and row['gpu'] == 0 and str(recovery.parent/'recovery_prior9_n128_01') == row['output']
        for pid in [row['pid'], row.get('child_pid')]:
            if pid is None:
                continue
            path = Path('/proc', str(pid))
            if path.exists():
                ticks = (path/'stat').read_text().rsplit(')', 1)[1].split()[19]
                native.append(process(pid, ticks))
    files.extend(receipts)
    record = dict(observed_at=datetime.now(timezone.utc).isoformat(), previous_goal_turn_classification='progress',
        previous_turn_evidence='5690290: finite source-admitted shared-target native entry and CPU waiting owner.',
        current_turn_classification='progress', window_search_and_cold_owners_and_native_handles_exited=True,
        current_recovery_owner=owner, current_recovery_state=state,
        selected_actual_source_sha256=cold['selected_policy_sha256'], selected_source_cold_summary=summary,
        recovery_selected_source_CPU_check=cpu_brief, actual_recovery_GPU_handles=native,
        downstream_CPU_only_waiters=waiting, held_union_files=len(held), all_held_inputs_unchanged=True,
        no_new_manual_GPU_launch=True, github_upload_performed=False,
        formal_cases_read=False, formal_acceptance=False, goal_complete=False,
        source_sha256={str(p): sha(p) for p in files})
    with (HERE/'CONTINUATION_EVIDENCE.json').open('x') as f:
        f.write(json.dumps(record, indent=2)+'\n')
    print(json.dumps(dict(completed_search_and_independent128_verified=True, current_recovery_state=state['state'],
        recovery_CPU_check=cpu_brief, actual_recovery_GPU_pids=[p['pid'] for p in native],
        held_union_files=len(held), all_unchanged=True, github_upload=False)))


if __name__ == '__main__':
    main()
