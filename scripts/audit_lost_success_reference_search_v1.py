#!/usr/bin/env python3
"""Search full recovery references at actual states from lost baseline successes.

Retrospective CPU diagnostic, not a runtime trigger or new physical execution.
Selection uses outcome labels only to study failures. All prior reference-memory
ticks are replayed, and speculative searches do not replace observed states.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quadsmash.rl.episode_trace_v4 import unpack_public
from scripts.landing_free_arm_reference_shooting_v1 import LandingFreeArmReferenceShootingMPCV1
from scripts.landing_support_mpc_v2 import LandingMPCSettingsV2
from scripts.whole_body_support_qp_v1 import PositionPDInterface, SupportSettings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--onset', type=Path, required=True)
    p.add_argument('--timing', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Preserve earlier evidence')
    read = lambda name: json.loads((a.run/name).read_text())
    if read('job.json')['status'] != 'SUCCEEDED':
        raise ValueError('Require completed native source')
    onset, timing = [json.loads(f.read_text()) for f in (a.onset, a.timing)]
    for data in (onset, timing):
        for name, digest in data['source_sha256'].items():
            path = Path(name)
            if not path.is_absolute():
                path = a.run/path
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('Selection evidence changed: '+str(path))
    lost = {r['case']:r for r in onset['cases'] if r['group']=='lost'}
    events = {r['case']:r['first_observed_nonfoot_s'] for r in timing['cases']}
    logs = read('landing_mpc_records.json'); selected = {}
    for case, data in lost.items():
        first = data['first_applied_qp_tick']
        if first is not None:
            selected[(case, first)] = 'first_actual_qp_application'
        event = events.get(case)
        if event is not None:
            candidates = [r['control_tick'] for r in logs if r['case']==case
                          and .01*r['control_tick'] <= event-.4]
            if candidates:
                selected.setdefault((case, max(candidates)), '400ms_before_first_nonfoot')
    native = read('native_command_inputs.json'); default = np.asarray(native['default_joint_position_rad'])
    interface = PositionPDInterface(np.asarray(native['joint_limits_rad']),
        np.r_[np.full(12,25.),np.full(6,80.)],np.r_[np.full(12,.5),np.full(6,4.)],
        np.r_[np.full(12,23.5),np.full(6,100.)],np.r_[np.full(12,30.),np.full(6,5.)],.65)
    cfg = SupportSettings(solver_eps=1e-6,residual_tolerance=3e-4,
        swing_foot_weight=1.,swing_foot_kp_s2=60.,swing_foot_kd_s=12.,swing_foot_clearance_m=.005,
        nonfoot_clearance_weight=5.,nonfoot_clearance_m=.05,
        nonfoot_clearance_kp_s2=60.,nonfoot_clearance_kd_s=15.)
    settings = LandingMPCSettingsV2(horizon_steps=40,minimum_support_feet=1,
        one_foot_max_com_distance_m=.20,landing_leg_posture_blend=.15,
        landing_arm_posture_blend=0.,executor_replay=True,predict_touchdown_contacts=True,
        contact_mode_fallback=True)
    mpc = LandingFreeArmReferenceShootingMPCV1(interface,qp_settings=cfg,
                                             settings=settings,arm_target_rate_rad_s=10.)
    mpc.set_posture_target(default)
    cache = {}; output = []
    files = [Path(__file__), a.onset, a.timing]
    for row in logs:
        case, tick = row['case'], row['control_tick']
        if case not in lost or tick > max(t for i,t in selected if i==case):
            continue
        if row.get('episode_index') != 0:
            raise ValueError('First-episode evidence only')
        if case not in cache:
            path = a.run/'fixed_policy_traces'/f'case_{case:03d}.npz'; files.append(path)
            with np.load(path,allow_pickle=False) as z:
                cache[case] = unpack_public(z['packed_public'])
        obs = cache[case][tick]
        if abs(float(obs[775])-.01*tick)>2e-6:
            raise ValueError('Clock mismatch')
        state = mpc.qp.model.state_from_public(obs,default)
        state.contacts = np.asarray(row['support_contacts'],dtype=bool)
        state.measured_force_world = np.asarray(row['measured_foot_force_world'])
        state.previous_command_target = np.asarray(row['previous_dispatched_target'])
        reference = mpc.admit_command_target(state,np.asarray(row['requested_student_joint_reference']))
        mpc.observe(state,reference,lane=case,control_tick=tick)
        if (case,tick) not in selected:
            continue
        started = time.perf_counter()
        plan = mpc.solve(state,reference,lane=case,control_tick=tick,force=True)
        search = plan['reference_shooting']; stages = plan.get('horizon',[])
        record = dict(case=case,tick=tick,selection=selected[(case,tick)],
            baseline_accepted=search['baseline_accepted'],search_accepted=bool(plan['accepted']),
            first_accepted=bool(stages and stages[0]['accepted']),
            first_target=stages[0].get('joint_position_target') if stages else None,
            rejection=plan.get('rejection'),terminal=plan.get('terminal_risk'),
            search=search,wall_seconds=time.perf_counter()-started)
        output.append(record)
        print(json.dumps({k:record[k] for k in ['case','tick','selection','baseline_accepted',
            'search_accepted','wall_seconds']}),flush=True)
    if len(output)!=len(selected):
        raise ValueError('Missing selected states')
    files += [a.run/n for n in ['job.json','native_command_inputs.json','landing_mpc_records.json']]
    files += [Path(module.__file__) for name,module in list(sys.modules.items())
              if name.startswith('scripts.') and getattr(module,'__file__',None)]
    result = dict(scope=__doc__,native_new_execution=False,source=str(a.run.resolve()),
        outcomes_used_only_for_retrospective_selection=True,contracts=mpc.contract(),
        summary=dict(samples=len(output),baseline_full_horizon=sum(r['baseline_accepted'] for r in output),
            search_full_horizon=sum(r['search_accepted'] for r in output),
            newly_feasible=[[r['case'],r['tick']] for r in output
                            if r['search_accepted'] and not r['baseline_accepted']],
            wall_seconds=sum(r['wall_seconds'] for r in output)),records=output,
        source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['summary']))


if __name__ == '__main__':
    main()
