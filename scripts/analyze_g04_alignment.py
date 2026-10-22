#!/usr/bin/env python3
"""CPU-only timing audit; optionally plot a future native reward_trace.csv.

Historical summaries cannot reconstruct missing COM velocities/reward terms.
Those fields are deliberately not invented from episode-level metrics.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trace', type=Path)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    source = ROOT / 'reports/G04/foot_ground_audit_v1/episodes.csv'
    capability = ROOT / 'reports/G04/capability_all_body_v2/metrics.json'
    best = json.loads(capability.read_text())['best']
    rows = list(csv.DictReader(source.open()))
    expert = [r for r in rows if 32 <= int(r['env_id']) < 64]
    comparisons = []
    for r in expert:
        index = (int(r['env_id']) - 32) % len(best)
        comparisons.append(dict(env_id=int(r['env_id']), source_parameter_env_id=best[index]['env_id'],
            formal_success=r['success'] == 'True', takeoff_time_s=float(r['takeoff_time_s']),
            landing_time_s=float(r['landing_time_s']), recovery_s=float(r['recovery_s']),
            reference_positive_bump_center_s=1.25,
            takeoff_minus_reference_center_s=float(r['takeoff_time_s']) - 1.25))
    report = dict(status='HISTORICAL_TIMING_ONLY', reference_note='1.25 s is the positive Gaussian bump center, not a measured feasible apex.',
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        capability_sha256=hashlib.sha256(capability.read_bytes()).hexdigest(), trajectories=comparisons,
        missing=['actual per-trajectory COM and vertical velocity', 'reward components for the engineering replay',
                 'reset COM required to reconstruct absolute reference'],
        conclusion='Timing mismatch motivates a controlled trace; it does not establish that the reference timing is physically impossible or rank controller returns.')
    if args.trace is not None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        trace = list(csv.DictReader(args.trace.open()))
        names = list(dict.fromkeys(r['group'] for r in trace))
        fig, axes = plt.subplots(len(names), 3, figsize=(15, 2.8 * len(names)), squeeze=False)
        selected_ids = {}
        for index, name in enumerate(names):
            selected_ids[name] = min(int(r['env_id']) for r in trace if r['group'] == name)
            rows = [r for r in trace if r['group'] == name and int(r['env_id']) == selected_ids[name]]
            t = [float(r['time_s']) for r in rows]
            for key in ('com_height_m', 'target_com_height_m'):
                axes[index, 0].plot(t, [float(r[key]) for r in rows], label=key)
            axes[index, 1].plot(t, [float(r['com_vertical_velocity_m_s']) for r in rows])
            for key in ('reward_height_tracking', 'reward_flight', 'reward_stable_recovery', 'reward_recovery_quality'):
                axes[index, 2].plot(t, [float(r[key]) for r in rows], label=key)
            for event in ('takeoff_time_s', 'landing_time_s'):
                for value in sorted({float(r[event]) for r in rows if float(r[event]) > 0.}):
                    axes[index, 0].axvline(value, color='gray', linewidth=.5, linestyle='--')
            for axis in axes[index]: axis.set_xlabel('episode time / s'); axis.grid(alpha=.2)
            axes[index, 0].set_ylabel(name + '\nheight / m'); axes[index, 1].set_ylabel('COM vertical speed / m/s')
            axes[index, 2].set_ylabel('weighted reward / policy step')
        axes[0, 0].legend(fontsize=7); axes[0, 2].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(args.output / 'paired_alignment.png', dpi=140); plt.close(fig)
        report.update(status='MEASURED_TRACE_ANALYSIS', trace_sha256=hashlib.sha256(args.trace.read_bytes()).hexdigest(),
                      plotted_env_ids=selected_ids, selection='first predetermined scenario per group; no outcome selection')
    (args.output / 'alignment.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'trajectories'}, indent=2))


if __name__ == '__main__': main()
