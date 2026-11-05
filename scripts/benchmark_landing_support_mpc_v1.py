#!/usr/bin/env python3
"""Benchmark the landing-support MPC on recorded native failure states.

This is a model/solver benchmark only. It never dispatches a command to Isaac
and therefore cannot be reported as improved recovery. It separates per-stage
QP feasibility from the stricter terminal-support acceptance gate.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from quadsmash.rl.episode_trace_v4 import unpack_public
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
from scripts.whole_body_support_qp_v1 import PositionPDInterface, SupportSettings
from scripts.landing_support_mpc_v1 import LandingSupportMPCV1, LandingMPCSettingsV1

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--timing-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--horizon-steps', type=int, default=10)
    parser.add_argument('--sample-offset-ticks', type=int, nargs='+', default=[0, 2])
    parser.add_argument('--allow-nonterminal-first-step', action='store_true')
    parser.add_argument('--swing-foot-weight', type=float, default=0.)
    parser.add_argument('--swing-foot-kp', type=float, default=80.)
    parser.add_argument('--swing-foot-kd', type=float, default=18.)
    parser.add_argument('--swing-foot-clearance', type=float, default=.025)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve earlier benchmark')
    if args.horizon_steps < 1 or args.horizon_steps > 40 or any(x < 0 for x in args.sample_offset_ticks):
        raise ValueError('Invalid horizon or sample offsets')
    source = args.evaluation.resolve()
    audit_path = args.timing_audit.resolve()
    audit = json.loads(audit_path.read_text())
    manifest = json.loads((source/'fixed_policy_traces/manifest.json').read_text())
    provenance = manifest['provenance']
    entries = {entry['source_training_index']: entry for entry in manifest['episodes']}
    limits = np.asarray(provenance['joint_limits_rad'], dtype=np.float64)
    default = np.asarray(provenance['default_joint_position_rad'], dtype=np.float64)
    interface = PositionPDInterface(
        limits, np.r_[np.full(12, 25.), np.full(6, 80.)],
        np.r_[np.full(12, .5), np.full(6, 4.)],
        np.r_[np.full(12, 23.5), np.full(6, 100.)],
        np.r_[np.full(12, 30.), np.full(6, 5.)], .65)
    qp_settings = SupportSettings(control_dt_s=.01, solver_eps=1e-6, residual_tolerance=3e-4,
        swing_foot_weight=args.swing_foot_weight, swing_foot_kp_s2=args.swing_foot_kp,
        swing_foot_kd_s=args.swing_foot_kd, swing_foot_clearance_m=args.swing_foot_clearance)
    mpc = LandingSupportMPCV1(interface, qp_settings=qp_settings,
        settings=LandingMPCSettingsV1(horizon_steps=args.horizon_steps))
    model = LoadedPinocchioModelV1()
    records = []
    for case in audit['cases']:
        index = int(case['training_index'])
        entry = entries[index]
        trace = source/'fixed_policy_traces'/entry['data_file']
        if sha(trace) != entry['data_sha256']:
            raise ValueError('Trace hash changed for TRAIN'+str(index))
        with np.load(trace, allow_pickle=False) as data:
            public = unpack_public(data['packed_public'])
            actions = data['dispatched_actions'].copy()
        clock = public[:, 775]
        landing_tick = int(np.argmin(np.abs(clock - case['landing_s'])))
        for offset in args.sample_offset_ticks:
            tick = min(landing_tick + offset, len(public) - 1)
            state = model.state_from_public(public[tick], default)
            joint_reference = limits[:, 0] + .5*(actions[tick] + 1.)*(limits[:, 1] - limits[:, 0])
            plan = mpc.solve(state, joint_reference, force=True)
            records.append(dict(training_index=index, offset_ticks=offset,
                clock_s=float(clock[tick]), support_feet=int(state.contacts.sum()),
                accepted=bool(plan['accepted']), solver_failure=bool(plan.get('solver_failure', False)),
                horizon_len=len(plan['horizon']),
                stage_feasible=sum(bool(x['accepted']) for x in plan['horizon']),
                stage_total=len(plan['horizon']), rejection=plan.get('rejection'),
                failed_stage=next((x for x in plan['horizon'] if not x.get('accepted')), None),
                terminal_risk=plan.get('terminal_risk'), runtime_ms=float(plan.get('wall_runtime_s', 0.)*1000.),
                first_action_available=bool(plan.get('horizon')) and bool(plan['horizon'][0].get('accepted'))))
    stage_count = sum(r['stage_total'] for r in records)
    stage_feasible = sum(r['stage_feasible'] for r in records)
    accepted = sum(r['accepted'] for r in records)
    first_available = sum(r['first_action_available'] for r in records)
    runtimes = np.asarray([r['runtime_ms'] for r in records], dtype=np.float64)
    result = dict(schema='landing-support-mpc-cpu-benchmark-v1', source=str(source),
        source_policy_sha256=json.loads((source/'INPUTS.json').read_text())['policy_sha256'],
        timing_audit=str(audit_path), timing_audit_sha256=sha(audit_path),
        source_trace_manifest_sha256=sha(source/'fixed_policy_traces/manifest.json'),
        cases=len(audit['cases']), samples=len(records), horizon_steps=args.horizon_steps,
        swing_foot_weight=args.swing_foot_weight, swing_foot_kp=args.swing_foot_kp,
        swing_foot_kd=args.swing_foot_kd, swing_foot_clearance=args.swing_foot_clearance,
        sample_offset_ticks=args.sample_offset_ticks, stage_qp_feasible=stage_feasible,
        stage_qp_total=stage_count, stage_qp_feasible_rate=stage_feasible/stage_count if stage_count else 0.,
        first_action_available=first_available, first_action_available_rate=first_available/len(records),
        terminal_acceptances=accepted, terminal_acceptance_rate=accepted/len(records),
        runtime_ms_quantiles=dict(median=float(np.median(runtimes)),p95=float(np.quantile(runtimes,.95)),
                                  maximum=float(np.max(runtimes))),
        records=records, mpc_contract=mpc.contract(),
        allow_nonterminal_first_step=bool(args.allow_nonterminal_first_step),
        native_execution=False, improved_recovery_proven=False, formal_acceptance=False,
        limitations=['Recorded state is reconstructed from public observation and authored Pinocchio model.',
                     'Contact mode is held fixed over the horizon; no impulse or new-foot contact is planned.',
                     'No MPC command was dispatched to Isaac; terminal acceptance is a model gate, not success.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:result[k] for k in ['cases','samples','horizon_steps','stage_qp_feasible_rate',
        'first_action_available_rate','terminal_acceptance_rate','runtime_ms_quantiles']}))


if __name__ == '__main__':
    main()
