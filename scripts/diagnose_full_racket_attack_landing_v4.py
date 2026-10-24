#!/usr/bin/env python3
"""Compare free-flight predictions with native landings and plan velocity targets."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quadsmash.physics.court_v4 import top_height
from quadsmash.physics.smash_feasibility import predict_flights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--generation', type=int, default=0)
    parser.add_argument('--env-id', type=int, required=True)
    parser.add_argument('--validate-runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh output directory')
    court_path = ROOT / 'configs/court_net_v4.json'
    court = json.loads(court_path.read_text())
    protocol = json.loads((ROOT / 'configs/g07_full_racket_eval_v4_r3.json').read_text())
    sources = [court_path, ROOT / 'configs/shuttle_physics_v2.json', Path(__file__),
               ROOT / 'quadsmash/physics/smash_feasibility.py', ROOT / 'quadsmash/physics/court_v4.py']

    def forecast(position, velocity):
        return predict_flights(position, velocity, dt=.0005, duration=2.,
            net_x=court['net_plane_x_m'], net_height=court['net_center_top_m'],
            net_half_width=court['net_width_m']/2,
            net_half_thickness=court['net_equivalent_thickness_m']/2,
            target_x=court['target_x_limits_m'], target_y=court['target_y_half_width_m'])

    def contacts(run, generation):
        name = run / f'contacts_{generation:03d}.json'
        sources.append(name)
        return json.loads(name.read_text())

    validation = []
    for run in args.validate_runs:
        source_path = run / 'source_hashes.json'
        sources.append(source_path)
        assert json.loads(source_path.read_text())[protocol['asset']['path']] == protocol['asset']['sha256']
        for path in sorted(run.glob('episodes_*.json')):
            suffix = path.stem.rsplit('_', 1)[1]
            if not suffix.isdigit():
                continue
            sources.append(path)
            events = contacts(run, int(suffix))
            for row in json.loads(path.read_text()):
                if not (row['airborne_string_hit'] and row['ground_seen'] and row['contact_count'] == 1
                        and not row['shuttle_net_contact'] and not row['body_ball_fault']):
                    continue
                event = next((e for e in events if e['env_id'] == row['env_id'] and e['first_racket_contact'] and e['string_contact']), None)
                if event is None:
                    continue
                predicted = forecast(event['ball_position_m'], event['ball_velocity_mps'])
                actual = np.array(row['first_landing_m'])
                validation.append(dict(run=run.name, generation=int(suffix), env_id=row['env_id'],
                    native_first_landing_m=actual.tolist(),
                    predicted_first_landing_m=predicted['landing_position'][0].tolist(),
                    landing_error_m=float(np.linalg.norm(predicted['landing_position'][0]-actual)),
                    flight_time_error_s=float(predicted['landing_time'][0]-(row['first_landing_time_s']-event['time_s']))))
    assert validation, 'Require observed native flights without a subsequent net impact'
    path = args.run / f'episodes_{args.generation:03d}.json'
    sources.append(path)
    row = next(r for r in json.loads(path.read_text()) if r['env_id'] == args.env_id)
    event = next(e for e in contacts(args.run, args.generation)
                 if e['env_id'] == args.env_id and e['first_racket_contact'] and e['string_contact'])
    assert row['released'] and np.isfinite(row['outgoing_velocity_mps']).all()
    position, velocity = np.array(event['ball_position_m']), np.array(row['outgoing_velocity_mps'])
    baseline = forecast(position, velocity)
    vx, vz = np.meshgrid(np.linspace(6.5, 10., 71), np.linspace(-4., -1., 61))
    candidates = np.column_stack((vx.ravel(), np.full(vx.size, velocity[1]), vz.ravel()))
    result = forecast(position, candidates)
    # Conservative planning clearance across the sphere's lateral footprint;
    # this buffer never becomes a task/acceptance gate.
    net_y = result['net_position'][:, 1]
    height = np.max(top_height(np.stack((net_y-.025, net_y, net_y+.025)), court), axis=0)
    exit_margin = result['conservative_net_exit_margin']-(height-court['net_center_top_m'])
    landing = result['landing_position']
    feasible = ((np.linalg.norm(candidates, axis=1) >= 8) & (exit_margin >= .005)
        & (landing[:, 0] >= court['target_x_limits_m'][0]) & (landing[:, 0] <= court['target_x_limits_m'][1])
        & (np.abs(landing[:, 1]) <= court['target_y_half_width_m']))
    proposal = None
    if feasible.any():
        cost = np.linalg.norm(candidates-velocity, axis=1)
        index = int(np.where(feasible, cost, np.inf).argmin())
        proposal = dict(outgoing_velocity_target_mps=candidates[index].tolist(),
            outgoing_speed_target_mps=float(np.linalg.norm(candidates[index])),
            delta_from_observed_outgoing_mps=(candidates[index]-velocity).tolist(),
            predicted_first_landing_m=landing[index].tolist(),
            conservative_net_exit_margin_m=float(exit_margin[index]))
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'flight_grid.npz', outgoing_velocity_mps=candidates,
        predicted_landing_m=landing, planning_feasible=feasible, conservative_exit_margin_m=exit_margin)
    report = dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='Unchanged drag/gravity free-flight forecast after an observed contact. Ball velocity proposals are policy-planning objectives, never simulator state writes. No new racket collision, joint tracking, body recovery or formal outcome is certified.',
        native_flight_validations=validation,
        maximum_native_landing_prediction_error_m=max(v['landing_error_m'] for v in validation),
        anchor_run=args.run.name, generation=args.generation, env_id=args.env_id,
        observed_outgoing_velocity_mps=velocity.tolist(),
        baseline_predicted_first_landing_m=baseline['landing_position'][0].tolist(),
        native_first_landing_observed=row['ground_seen'],
        planning_clearance_buffer_m=.005, unchanged_target_x_m=court['target_x_limits_m'],
        unchanged_target_y_half_width_m=court['target_y_half_width_m'],
        sampled_velocity_targets=len(candidates), feasible_free_flight_targets=int(feasible.sum()),
        nearest_free_flight_proposal=proposal, changed_state_dynamics=False,
        all_original_physical_and_task_gates_unchanged=True, formal_acceptance=False,
        source_sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (args.output/'REPORT.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','native_flight_validations']}), flush=True)


if __name__ == '__main__':
    main()
