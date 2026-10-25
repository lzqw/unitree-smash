"""CPU-only resource-gate checks; synthetic inputs never prove native performance."""
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('formal_queue_under_check', HERE / 'run.py')
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def main():
    output = HERE / 'PRELAUNCH_CHECKS.json'
    assert not output.exists(), 'Preserve previous check receipts'
    assert not queue.STATE.exists() and not queue.RESULTS.exists()
    protocol = queue.core.read(queue.PROTOCOL)
    report_path = queue.REPORTS / 'residual_iter32_fixed_diagnostic_20261025/REPORT.json'
    actual = queue.core.read(report_path)['independently_recomputed_fixed_TRAIN128']
    checks = []

    def check(name, summary, expected, config=protocol):
        actual_decision = queue.decide(summary, config)
        assert actual_decision == expected, (name, actual_decision)
        checks.append(dict(name=name, result=actual_decision, passed=True))

    check('actual_completed_iter32_fixed_TRAIN128_must_skip', actual,
          'SKIPPED_TRAIN_BELOW_ORIGINAL_THRESHOLDS')
    # Explicit guard-unit fixtures, with rates derived from integer counts.
    # They do not stand in for a native rollout or an acceptance result.
    fixture = dict(episodes=128, complete_success_rate=64/128,
                   airborne_contact_rate=90/128, conditional_recovery_rate=77/90,
                   numerical_failures=0)
    check('synthetic_all_original_rates_qualify', fixture, 'RUN_ORIGINAL_FORMAL_AB')
    for key, value in [('complete_success_rate', 63/128),
                       ('airborne_contact_rate', 89/128),
                       ('conditional_recovery_rate', 76/90)]:
        changed = dict(fixture, **{key: value})
        check('synthetic_below_' + key, changed, 'SKIPPED_TRAIN_BELOW_ORIGINAL_THRESHOLDS')
    check('synthetic_nonzero_numerical_failure_stops', dict(fixture, numerical_failures=1),
          'STOP_FOR_NUMERICAL_FAILURE')
    for key, values in [('episodes', [127, 129, None]),
                         ('numerical_failures', [-1, True, .5]),
                         ('complete_success_rate', [float('nan'), float('inf'), -0.1, 1.1, True, '0.5']),
                         ('airborne_contact_rate', [float('nan'), -.1, 1.1]),
                         ('conditional_recovery_rate', [float('nan'), -.1, 1.1])]:
        for index, value in enumerate(values):
            try:
                queue.decide(dict(fixture, **{key: value}), protocol)
            except ValueError:
                checks.append(dict(name=f'synthetic_invalid_{key}_{index}_rejected', passed=True))
            else:
                raise AssertionError((key, index))
    missing = dict(fixture)
    del missing['episodes']
    try:
        queue.decide(missing, protocol)
    except KeyError:
        checks.append(dict(name='synthetic_missing_episode_count_rejected', passed=True))
    else:
        raise AssertionError('Missing episode count was accepted')
    changed_protocol = deepcopy(protocol)
    changed_protocol['thresholds']['complete_success_rate'] = .49
    try:
        queue.decide(fixture, changed_protocol)
    except ValueError:
        checks.append(dict(name='synthetic_lowered_formal_threshold_rejected', passed=True))
    else:
        raise AssertionError('Changed formal thresholds were accepted')

    commands = {panel: queue.formal_command(panel) for panel in ['A', 'B']}
    for panel, command in commands.items():
        assert command[0] == queue.PYTHON
        for option, value in [('--goal', 'G07'),
                ('--budget-window', 'g07-full-racket-unlimited-20261024'),
                ('--max-seconds', '2700'), ('--env-steps-cap', '160000'),
                ('--checkpoint', str(queue.POLICY)), ('--panel', panel),
                ('--protocol', str(queue.PROTOCOL)),
                ('--policy-lock', str(queue.RESULTS / 'lock/LOCK.json'))]:
            assert command[command.index(option) + 1] == value
        assert [command[i+1] for i, arg in enumerate(command) if arg == '--output'] == [str(queue.RESULTS / panel)] * 2
        assert 'scripts/evaluate_full_racket_v4.py' in command and '--headless' in command
        checks.append(dict(name=f'formal_{panel}_command_original_inputs_and_caps', passed=True))
    assert queue.core.live(queue.PARENT_PID, 'residual_resize2048_queue01/run.py')
    assert queue.native.hashes() == queue.core.read(queue.PARENT / 'state.json')['source_hashes']
    assert not queue.STATE.exists() and not queue.RESULTS.exists()
    queue.core.atomic(output, dict(created_at=datetime.now(timezone.utc).isoformat(),
        status='CPU_RESOURCE_GATE_CHECKS_PASSED', checks=checks, total_checks=len(checks),
        source_sha256=queue.core.sha(HERE / 'run.py'), checker_sha256=queue.core.sha(Path(__file__)),
        actual_completed_diagnostic=dict(path=str(report_path.relative_to(queue.ROOT)),
            sha256=queue.core.sha(report_path), episodes=actual['episodes'],
            airborne_contacts=actual['airborne_contacts'],
            conditional_recovery_count=actual['conditional_recovery_count'],
            complete=actual['complete'], numerical_failures=actual['numerical_failures']),
        synthetic_fixture_scope='Pure decision-unit inputs, never native performance or policy acceptance evidence.',
        verified_live_parent_pid=queue.PARENT_PID, current_parent_sources_unchanged=True,
        new_GPU_jobs_started=False, formal_cases_read=False, formal_acceptance=False,
        goal_complete=False, planned_commands=commands))
    print(json.dumps(dict(status='CPU_RESOURCE_GATE_CHECKS_PASSED', total_checks=len(checks),
                         receipt=str(output), new_GPU_jobs_started=False)))


if __name__ == '__main__':
    main()
