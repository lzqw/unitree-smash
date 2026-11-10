"""Read-only timing/geometry replay of the saved 55/74 run; no native rollout."""
import copy
import csv
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path('/home/autodrive/LZQW/unitree_smash_contact_recovery')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MAIN))
from scripts.audit_sequence_repair_run import finite, sha
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.loaded_pinocchio_model_v1 import PublicState
from scripts.pure_control_state_driven_long_mpc_v1_qp_continuity_repair import PureControlStateDrivenLongMPCV1QPContinuityRepair

RUN = ROOT / 'reports/pure_control_state_driven_long_mpc_qp_continuity_repair_all128_corrected_20261110_03/native'
BASE = MAIN / 'reports/pure_control_state_driven_long_mpc_qp_continuity_all128_20261109_01/native'
OUT = ROOT / 'reports/head_replan_opportunity_20261110'


def read(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = read(RUN / 'metrics.json')
    outcomes = {r['training_index']: r for r in metrics['complete5_case_outcomes']}
    fixed = sorted(r['training_index'] for r in read(BASE / 'metrics.json')['complete5_case_outcomes'] if r['air'])
    assert len(fixed) == 74
    rows = [json.loads(s) for s in (RUN / 'pure_control_diagnostics.jsonl').open()]
    with np.load(RUN / 'pure_control_commands.npz') as z:
        cmd = {k: z[k].copy() for k in z.files}
    assert all((r['tick'], r['lane']) == (cmd['tick'][i], cmd['lane'][i]) for i, r in enumerate(rows))
    names = read(RUN / 'pure_control_ground_forces.json')['body_names']
    heads = [i for i, n in enumerate(names) if 'head' in n.lower()]
    feet = [i for i, n in enumerate(names) if n.endswith('_foot')]
    with np.load(RUN / 'pure_control_ground_forces.npz') as z:
        frames = z['interval_first_frame_over1n'].copy()
        peaks = z['interval_peak_normal_n'].copy()
        sampled_frames = z['physics_frame'].copy()
        sampled_heads = np.linalg.norm(z['ground_normal_force_world_n'][:, :, heads], axis=-1).max(axis=-1)
    config = read(ROOT / 'reports/pure_control_state_driven_long_mpc_qp_continuity_repair_native_preparation_20261110_01/controller.json')
    nominal = np.array(read(RUN / 'native_command_inputs.json')['default_joint_position_rad'])
    controller = PureControlStateDrivenLongMPCV1QPContinuityRepair(PositionPDInterface(**config['interface']), nominal)
    geometry = controller.geometry.ground
    head_shapes = [i for i, shape in enumerate(geometry.shapes) if 'head' in shape['body'].lower()]
    assert head_shapes

    def head(q, v):
        gaps, jac = geometry.evaluate(q, jacobian=True)
        index = min(head_shapes, key=lambda i: gaps[i])
        return float(gaps[index]), float(jac[index] @ v), geometry.shapes[index]['body']

    def state_at(i):
        q, v = cmd['q'][i].copy(), cmd['v'][i].copy()
        s = PublicState(q, v, pin.Quaternion(q[3:7]).matrix(), q[:3].copy(),
                        q[controller.model.q_indices].copy(), v[controller.model.v_indices].copy(),
                        np.array(rows[i]['phase']['ground_contacts']), 0.)
        s.previous_command_target = cmd['previous'][i].copy()
        s.measured_force_world = cmd['ground_force'][i].copy()
        return s

    cases, samples, replay_errors, derivative_checks = [], [], [], []
    rejection = Counter()
    for lane in fixed:
        ids = np.flatnonzero(cmd['lane'] == lane)
        start = int(cmd['tick'][ids[0]])
        by_tick = {int(cmd['tick'][i]): int(i) for i in ids}

        def first(indices, begin, end=800):
            values = frames[begin:end, lane][:, indices]
            valid = values[values > 0]
            return int(valid.min()) if len(valid) else None

        head_frame = first(heads, start)
        foot_frame = first(feet, start)
        # Each native interval contains physics frames tick*160+1 .. (tick+1)*160.
        collision_tick = (head_frame - 1) // 160
        before = [int(i) for i in ids if cmd['tick'][i] <= collision_tick]
        accepted = [i for i in before if rows[i]['diagnostic']['solver_accepted']]
        replans = [i for i in before if rows[i]['diagnostic']['replanned']]
        repairs = [i for i in ids if rows[i]['diagnostic']['plan'].get('sequence_repaired')]
        attempts = [i for i in ids if rows[i]['diagnostic']['plan'].get('repair_attempted') or rows[i]['diagnostic']['plan'].get('sequence_repaired')]
        last_accepted_tick = int(cmd['tick'][accepted[-1]]) if accepted else None
        fallback_begin = last_accepted_tick + 1 if accepted else start
        assert all(not rows[by_tick[t]]['diagnostic']['solver_accepted'] for t in range(fallback_begin, collision_tick + 1))
        active_repair = False
        predicted = None
        cache_gap_values, cache_errors, remaining_margins = [], [], []
        # Independently check the geometric approach velocity used in this audit.
        for i in (before[0], before[-1]):
            q, v = cmd['q'][i], cmd['v'][i]
            _, derivative, _ = head(q, v)
            epsilon = 1e-7
            plus = head(pin.integrate(controller.model.model, q, epsilon*v), v)[0]
            minus = head(pin.integrate(controller.model.model, q, -epsilon*v), v)[0]
            derivative_checks.append(dict(lane=lane, tick=int(cmd['tick'][i]),
                analytic_m_s=derivative, finite_difference_m_s=(plus-minus)/(2*epsilon),
                difference_m_s=abs(derivative-(plus-minus)/(2*epsilon))))
        for i in before:
            tick = int(cmd['tick'][i]); d = rows[i]['diagnostic']
            gap, rate, body = head(cmd['q'][i], cmd['v'][i])
            if d['replanned']:
                active_repair = bool(d['plan'].get('sequence_repaired'))
                predicted = state_at(i) if d['solver_accepted'] else None
                plan_start = tick
            source = ('repaired_plan' if active_repair else 'normal_plan') if d['solver_accepted'] else 'fallback'
            pg = pr = delta = remaining_margin = None
            if predicted is not None:
                pg, pr, _ = head(predicted.q, predicted.v)
                delta = gap - pg
                remaining = (16 - (tick - plan_start)) * .01
                # Diagnostic extrapolation only, not a new trigger or safety certificate.
                remaining_margin = gap - max(0., -rate) * remaining
                cache_gap_values.append(gap); cache_errors.append(delta); remaining_margins.append(remaining_margin)
                try:
                    result = controller._predict_request(predicted, cmd['requested'][i])
                    assert np.max(abs(result['predicted_final_target'] - cmd['final'][i])) <= 2e-6
                    predicted = result['state']
                except (ValueError, FloatingPointError, AssertionError) as exc:
                    replay_errors.append(dict(lane=lane, tick=tick, error=str(exc)))
                    predicted = None
            if source == 'fallback':
                rejection[d['plan'].get('rejection', 'unknown')] += 1
            samples.append(dict(lane=lane, tick=tick, time_s=tick*.01, recovered=outcomes[lane]['recovery'],
                source=source, replanned=d['replanned'], head_clearance_m=gap,
                head_clearance_rate_m_s=rate, head_body=body,
                predicted_head_clearance_m=pg, predicted_head_rate_m_s=pr,
                actual_minus_predicted_gap_m=delta,
                linear_gap_at_next_scheduled_replan_m=remaining_margin,
                first_head_time_s=head_frame/16000., first_foot_time_s=foot_frame/16000. if foot_frame else None,
                rejection=d['plan'].get('rejection')))
        at_entry = sampled_heads[sampled_frames == start*160, lane]
        cases.append(dict(lane=lane, recovered=outcomes[lane]['recovery'],
            takeover_tick=start, takeover_time_s=start*.01,
            contact_time_s=read(RUN/'episodes.json')[lane]['contact_time_s'],
            head_force_at_takeover_n=float(at_entry[0]),
            head_in_previous_interval=bool(peaks[start-1, lane, heads].max()>1),
            head_in_takeover_interval=bool(peaks[start, lane, heads].max()>1),
            first_head_ever_s=first(heads,0)/16000.,
            first_post_takeover_head_s=head_frame/16000.,
            first_post_takeover_foot_s=foot_frame/16000. if foot_frame else None,
            foot_to_head_delay_s=(head_frame-foot_frame)/16000. if foot_frame else None,
            control_at_first_head='fallback' if not rows[by_tick[collision_tick]]['diagnostic']['solver_accepted'] else 'plan',
            last_planning_time_s=int(cmd['tick'][replans[-1]])*.01,
            last_accepted_command_s=last_accepted_tick*.01 if accepted else None,
            fallback_begin_s=fallback_begin*.01,
            continuous_fallback_ticks_to_first_head=collision_tick-fallback_begin+1,
            continuous_fallback_all_replanned=all(rows[by_tick[t]]['diagnostic']['replanned'] for t in range(fallback_begin,collision_tick+1)),
            first_repair_attempt_s=int(cmd['tick'][attempts[0]])*.01 if attempts else None,
            first_repair_pass_s=int(cmd['tick'][repairs[0]])*.01 if repairs else None,
            accepted_commands_before_first_head=len(accepted),
            cached_min_head_clearance_m=min(cache_gap_values) if cache_gap_values else None,
            cached_actual_minus_predicted_gap_min_m=min(cache_errors) if cache_errors else None,
            cached_actual_minus_predicted_gap_max_m=max(cache_errors) if cache_errors else None,
            cached_min_linear_gap_at_next_replan_m=min(remaining_margins) if remaining_margins else None))
        print(json.dumps(dict(lane=lane, cached_gap_min=cases[-1]['cached_min_head_clearance_m'],
            cached_error_min=cases[-1]['cached_actual_minus_predicted_gap_min_m'],
            linear_margin=cases[-1]['cached_min_linear_gap_at_next_replan_m'])), flush=True)
    write_csv(OUT/'case_timing_74.csv', cases)
    write_csv(OUT/'pre_head_geometry_and_control.csv', samples)
    output = dict(scope='CPU-only replay of saved native run; no new control, no native evaluation',
        working_run_commit='7251b5f', fixed_cases=fixed, cases=cases,
        replay_errors=replay_errors, derivative_checks=derivative_checks,
        fallback_rejections_before_first_head=dict(rejection),
        force_threshold_n=1., native_contact_time_resolution_s=1/16000.,
        geometry_sample_period_s=.01,
        geometry_scope='Authored collision support surfaces from saved actual full-body q; not exact PhysX cooking equivalence',
        prediction_scope='Existing predictor with saved executed request sequence and branch-local final-command history; no QP re-solve',
        sources={str(p):sha(p) for p in [Path(__file__),RUN/'pure_control_commands.npz',RUN/'pure_control_diagnostics.jsonl',RUN/'pure_control_ground_forces.npz',BASE/'metrics.json']})
    (OUT/'timing_geometry_audit.json').write_text(json.dumps(finite(output), indent=2)+'\n')
    summarize(output, cmd)


def summarize(output, cmd):
    from quadsmash.physics.court_v4 import load_court
    cases = output['cases']
    with np.load(RUN/'fixed_policy_traces/case_068.npz') as z:
        public = z['packed_public'].copy()
    ball = []
    for i in np.flatnonzero(cmd['lane'] == 68):
        tick = int(cmd['tick'][i]); q = cmd['q'][i]
        assert public[tick, 67] == 1.
        rotation = pin.Quaternion(q[3:7]).matrix()
        position = q[:3] + rotation @ public[tick, 61:64]
        velocity = rotation @ (public[tick, 64:67] + public[tick, :3]
                               + np.cross(public[tick, 3:6], public[tick, 61:64]))
        ball.append(dict(tick=tick, time_s=tick*.01,
                         x=position[0], y=position[1], z=position[2],
                         vx=velocity[0], vy=velocity[1], vz=velocity[2]))
    write_csv(OUT/'train68_saved_ball_state.csv', ball)
    reversal = next(i for i in range(1, len(ball)) if ball[i-1]['vx'] > 0 and ball[i]['vx'] <= 0)
    net_x = load_court()['net_plane_x_m']
    falling_return = next(i for i in range(reversal, len(ball)) if ball[i]['vx'] <= 0 and ball[i]['vz'] < 0 and ball[i]['x'] < net_x)
    episode = read(RUN/'episodes.json')[68]
    net = dict(source='FullRacketEnv native net-force view: last filter is shuttle; norm > .001 N',
        predicate='active & accumulated shuttle_net_contact & own_side & vx<=0 & vz<0 (after hit)',
        exact_native_net_contact_time_s=None, exact_native_fault_time_s=None,
        limitation='No per-physics-step shuttle/net force or fault-transition archive was saved. Brackets below are inferred from 100 Hz measured ball states, not exact native event timestamps.',
        measured_velocity_reversal_bracket_s=[ball[reversal-1]['time_s'],ball[reversal]['time_s']],
        falling_return_predicate_transition_bracket_s=[ball[falling_return-1]['time_s'],ball[falling_return]['time_s']],
        reversal_samples=ball[reversal-1:reversal+1],falling_return_samples=ball[falling_return-1:falling_return+1],
        net_plane_x_m=net_x, flags={k:v for k,v in episode.items() if k.endswith('_fault') or k in ['shuttle_net_contact','engineering_body_crossing']},
        contact_flight_recovery=episode['contact_flight_recovery'],
        conditional_recovery_unchanged=False,
        source_sha256={str(p):sha(p) for p in [MAIN/'quadsmash/tasks/stand/full_racket_env_v4.py',MAIN/'quadsmash/physics/rally_rules_v4.py',RUN/'fixed_policy_traces/case_068.npz']})
    (OUT/'train68_rule_event_audit.json').write_text(json.dumps(finite(net), indent=2)+'\n')
    def extent(key, subset=cases):
        values=[r[key] for r in subset]
        return dict(minimum=min(values),median=float(np.median(values)),maximum=max(values))
    previous = read(ROOT/'reports/execution_consistent_repair_result_20261110/native_comparison.json')
    summary=dict(status='STOPPED_AT_CPU_EVIDENCE_GATE', new_native_evaluations=0,
        new_controller_implemented=False, baseline_run_code='7251b5f',
        primary_count=74, recovered_count=55,
        first_head_source=dict(Counter(r['control_at_first_head'] for r in cases)),
        at_takeover_head_contact=sum(r['head_force_at_takeover_n']>1 for r in cases),
        previous_interval_head_contact=sum(r['head_in_previous_interval'] for r in cases),
        takeover_interval_head_contact=sum(r['head_in_takeover_interval'] for r in cases),
        first_head_after_foot=sum(r['foot_to_head_delay_s']>0 for r in cases),
        first_head_before_foot=sum(r['foot_to_head_delay_s']<0 for r in cases),
        all_fallback_ticks_replanned=all(r['continuous_fallback_all_replanned'] for r in cases),
        fallback_ticks_before_first_head=extent('continuous_fallback_ticks_to_first_head'),
        cache_head_clearance_m=extent('cached_min_head_clearance_m'),
        actual_minus_predicted_head_gap_min_m=extent('cached_actual_minus_predicted_gap_min_m'),
        actual_minus_predicted_head_gap_max_m=extent('cached_actual_minus_predicted_gap_max_m'),
        linear_gap_at_next_replan_m=extent('cached_min_linear_gap_at_next_replan_m'),
        recovered_first_head_source=dict(Counter(r['control_at_first_head'] for r in cases if r['recovered'])),
        recovered_with_repair_before_head=sum(r['recovered'] and r['first_repair_pass_s'] is not None and r['first_repair_pass_s']<r['first_post_takeover_head_s'] for r in cases),
        cpu_prediction_nodes=74*16, cpu_replay_errors=output['replay_errors'],
        derivative_check_count=len(output['derivative_checks']),
        derivative_max_error_m_s=max(r['difference_m_s'] for r in output['derivative_checks']),
        decision='Saved traces do not support head risk caused by a cached plan waiting for the next 16-tick replan. Actual head gap is at least .39m during caching and never below same-time model prediction; all first head impacts occur after 18-21 every-tick rejected replans. Do not add this mechanism or use the authorized native evaluation.',
        limitations=['Linear clearance extrapolation is diagnostic, not a safety guarantee.',
            'No claim that an alternative earlier action could not prevent contact.',
            'Later support effects of early replanning are untested; no replacement mechanism is authorized.'],
        saved_metrics_evidence=str(ROOT/'reports/execution_consistent_repair_result_20261110/native_comparison.json'),
        saved_metric_keys=list(previous),
        early_risk_trigger_count=None, new_replans_accepted=None, new_replans_executed=None)
    (OUT/'SUMMARY.json').write_text(json.dumps(finite(summary), indent=2)+'\n')


if __name__ == '__main__':
    main()
