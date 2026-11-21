"""Pause the registered researcher on sustained deterioration in completed native episodes."""
import argparse
from collections import deque
import fcntl
import json
from math import hypot, isfinite
from pathlib import Path
from statistics import mean
import time

import psutil

from scripts.report_supermimic_research_curve import FirstEpisodeReturns, read_new


def case_evidence(observed, baseline):
    """Preserve native scoring flags for the same lane's first episode."""
    def scores(row):
        flight = row.get('native_flight', {})
        velocity = flight.get('outgoing_velocity_mps')
        if 'release_time_s' in flight and (flight['release_time_s'] is None or not flight['release_time_s'] >= 0):
            velocity = None
        speed = None
        if velocity is not None and len(velocity) == 3 and all(v is not None and isfinite(v) for v in velocity):
            speed = hypot(*velocity)
        return dict(native_return=row['native_episode_return'], body_ground=row['body_ground_seen'],
                    qualified_contact=row['qualified_airborne_string_contact'],
                    ball_side=row['ball_side_complete_except_recovery'], complete5=row['complete5'],
                    episode_length_steps=row.get('native_episode_length_steps'), timeout=row.get('episode_timeout'),
                    physical_recovery=row.get('physical_recovery_success'), release_speed_mps=speed,
                    failed_criteria=row.get('rejection_reasons'))

    return dict(case=observed['case_id'], lane=observed['env_id'], episode=observed['episode_index'],
                observed=observed['native_episode_return'], terminal_behavior=observed.get('behavior_at_terminal'),
                observed_scores=scores(observed), baseline_scores=scores(baseline) if baseline else None,
                paired_delta=observed['native_episode_return']-baseline['native_episode_return'] if baseline else None)


def attach_baseline_flights(baseline, path):
    # Older terminal summaries omit the latched velocity, but keep a linked
    # raw episode file. Use only first episodes that reproduce the scored Return.
    raw, _ = read_new(path, 0)
    for row in raw:
        key = row['case_id'], row['env_id']
        if row['episode_index'] == 0 and key in baseline:
            scored = baseline[key]
            if row.get('native_episode_return') == scored['native_episode_return'] and row.get('released'):
                scored['native_flight'] = dict(outgoing_velocity_mps=row.get('outgoing_velocity_mps'))


class ReturnTrend:
    def __init__(self, baseline, minimum_cases=4):
        self.baseline = baseline
        self.minimum_cases = minimum_cases
        self.paired_sum = self.paired_cases = 0
        self.pending = []
        self.recent = deque(maxlen=128)
        self.cohorts = deque(maxlen=3)

    def observe(self, episodes, tick):
        self.pending.extend(episodes)
        self.recent.extend(e['native_episode_return'] for e in episodes)
        if len({e['env_id'] for e in self.pending}) < self.minimum_cases:
            return None
        rows, self.pending = self.pending, []
        pairs = [(e, self.baseline[(e['case_id'], e['env_id'])]) for e in rows
                 if e['episode_index'] == 0 and (e['case_id'], e['env_id']) in self.baseline]
        if any(e['scoring_schema'] != b['scoring_schema'] for e, b in pairs):
            raise ValueError('Inconsistent paired native scores')
        deltas = [e['native_episode_return'] - b['native_episode_return'] for e,b in pairs]
        self.paired_cases += len(deltas)
        self.paired_sum += sum(deltas)
        fields = ('body_ground_seen', 'qualified_airborne_string_contact', 'ball_side_complete_except_recovery', 'complete5')
        current = dict(tick=tick, count=len(rows), mean=mean(e['native_episode_return'] for e in rows),
                       return_mean_128=mean(self.recent), paired_count=len(pairs),
                       paired_delta=mean(deltas) if deltas else None,
                       cumulative_paired_delta=self.paired_sum/self.paired_cases if self.paired_cases else None,
                       paired_net={k:sum(int(e[k])-int(b[k]) for e,b in pairs) for k in fields})
        matched = {(e['case_id'], e['env_id']): b for e,b in pairs}
        current['cases'] = [case_evidence(e, matched.get((e['case_id'], e['env_id']))
                                          if e['episode_index'] == 0 else None) for e in rows]
        self.cohorts.append(current)
        history = list(self.cohorts)
        reason = None
        if len(history) >= 2:
            a,b = history[-2:]
            if (all(c['paired_count'] == c['count'] for c in (a,b))
                    and b['paired_delta'] < a['paired_delta'] < 0
                    and b['return_mean_128'] < a['return_mean_128'] and b['mean'] < a['mean']):
                reason = 'paired_first_episode_loss_worsened'
        if reason is None and len(history) == 3 and all(c['paired_count'] == c['count'] for c in history):
            # Successive case mixes can have different baseline difficulty. Detect a
            # sustained decline in both recorded curves even if a cohort mean rebounds.
            if all(b['paired_delta'] < 0 and b['return_mean_128'] < a['return_mean_128']
                   and b['cumulative_paired_delta'] < a['cumulative_paired_delta']
                   for a,b in zip(history, history[1:])):
                reason = 'consecutive_native_and_paired_curve_declines'
        if len(history) == 3 and any(c['paired_count'] != c['count'] for c in history):
            if all(b['return_mean_128'] < a['return_mean_128'] and b['mean'] < a['mean']
                   for a,b in zip(history, history[1:])):
                reason = 'consecutive_mixed_episode_return_declines'
        return dict(reason=reason, cohorts=history,
                    cases=current['cases'])


def monitor(arm, trainer):
    train = arm/'train'
    cmd = trainer.cmdline()
    if ('scripts/train_supermimic_physical_block.py' not in cmd or Path(cmd[cmd.index('--output')+1]).resolve()!=train):
        raise ValueError('Process and arm do not match')
    registration = json.loads((arm/'registration.json').read_text())
    manifest_path = train/'native_transitions/manifest.json'
    while not manifest_path.exists() and trainer.is_running():
        time.sleep(5)
    if not manifest_path.exists():
        return
    baseline = FirstEpisodeReturns(arm, registration, json.loads(manifest_path.read_text())).baseline
    attach_baseline_flights(baseline, Path(registration['baseline_path']).with_name('episodes.jsonl'))
    trend = ReturnTrend(baseline)
    episode_offset = history_offset = consumed = 0
    episodes = []
    decisions = arm/'native_return_pauses.jsonl'
    previous, _ = read_new(decisions, 0)
    acknowledged = max((r['through_episode'] for r in previous), default=0)
    while trainer.is_running() and trainer.status() != psutil.STATUS_ZOMBIE:
        history, history_offset = read_new(train/'training_history.jsonl', history_offset)
        if history:
            new, episode_offset = read_new(train/'native_episodes.jsonl', episode_offset)
            episodes.extend(new)
        for row in history:
            end = row['completed_native_episodes']
            if end == consumed:
                continue
            if not consumed < end <= len(episodes):
                raise ValueError('Trainer history and completed episode counts disagree')
            decision = trend.observe(episodes[consumed:end], row['native_ticks'])
            consumed = end
            if decision is None:
                continue
            if consumed == acknowledged:
                trend.cohorts.clear()
            if not decision['reason'] or consumed <= acknowledged:
                continue
            trainer.suspend()
            decision.update(arm=registration['arm'], trainer_pid=trainer.pid, timestamp=time.time(),
                            through_episode=end, accepted_id=row['accepted_id'])
            with decisions.open('a') as stream:
                stream.write(json.dumps(decision,allow_nan=False)+'\n')
            print(json.dumps(decision),flush=True)
            acknowledged = consumed
            trend.cohorts.clear()
            break
        time.sleep(5)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision',type=Path,required=True)
    parser.add_argument('--supervisor-pid',type=int,required=True)
    args=parser.parse_args()
    root=args.revision.resolve()
    supervisor=psutil.Process(args.supervisor_pid)
    with (root/'native_return_watch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while supervisor.is_running() and supervisor.status()!=psutil.STATUS_ZOMBIE:
            for child in supervisor.children():
                try:
                    cmd=child.cmdline()
                    if 'scripts/train_supermimic_physical_block.py' not in cmd:
                        continue
                    arm=Path(cmd[cmd.index('--output')+1]).resolve().parent
                    if arm.parent==root:
                        print(f'Monitoring {arm} PID {child.pid}',flush=True)
                        monitor(arm,child)
                except psutil.NoSuchProcess:
                    continue
            time.sleep(5)
