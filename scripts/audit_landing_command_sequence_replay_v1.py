#!/usr/bin/env python3
"""Offline replay of actual command sequences; not a deployable predictor.

Compare held initial support with recorded future force-contact flags. The
latter is explicitly an oracle diagnostic, never a causal controller input.
Both reuse the existing implicit dynamics, without impact or sliding physics.
Native episode termination is respected and no missing future is extrapolated.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quadsmash.rl.episode_trace_v4 import unpack_public
from scripts.implicit_support_prediction_v1 import predict_implicit_support
from scripts.landing_command_history_mpc_v1 import LandingCommandHistoryMPCV1
from scripts.landing_support_mpc_v2 import LandingMPCSettingsV2
from scripts.whole_body_support_qp_v1 import PositionPDInterface, SupportSettings


def errors(predicted, actual, model):
    rp = pin.Quaternion(predicted.q[3:7]).matrix()
    ra = pin.Quaternion(actual.q[3:7]).matrix()
    return dict(root_height_predicted_m=float(predicted.q[2]),
        root_height_actual_m=float(actual.q[2]),
        root_position_error_m=float(np.linalg.norm(predicted.q[:3]-actual.q[:3])),
        root_attitude_error_rad=float(np.linalg.norm(pin.log3(ra.T@rp))),
        world_linear_velocity_error_mps=float(np.linalg.norm(rp@predicted.v[:3]-ra@actual.v[:3])),
        world_angular_velocity_error_rad_s=float(np.linalg.norm(rp@predicted.v[3:6]-ra@actual.v[3:6])),
        joint_error_max_rad=float(np.max(abs(predicted.q[model.q_indices]-actual.joint_position))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise ValueError('Preserve existing evidence')
    read = lambda name: json.loads((a.run/name).read_text())
    if read('job.json')['status'] != 'SUCCEEDED': raise ValueError('Completed native run required')
    native = read('native_command_inputs.json')
    default = np.asarray(native['default_joint_position_rad'])
    interface = PositionPDInterface(np.asarray(native['joint_limits_rad']),
        np.r_[np.full(12,25.),np.full(6,80.)],np.r_[np.full(12,.5),np.full(6,4.)],
        np.r_[np.full(12,23.5),np.full(6,100.)],np.r_[np.full(12,30.),np.full(6,5.)],.65)
    mpc = LandingCommandHistoryMPCV1(interface, qp_settings=SupportSettings(),
        settings=LandingMPCSettingsV2(executor_replay=True), arm_target_rate_rad_s=10.)
    model = mpc.qp.model
    records = {(r['case'],r['control_tick']): r for r in read('landing_mpc_records.json')
               if r['episode_index'] == 0}
    # First accepted windows plus pre-failure / later accepted windows.
    selections = [(12,318),(18,310),(18,330),(50,322),(50,335),(67,390),(67,500),(98,385)]
    sources = [a.run/n for n in ['job.json','native_command_inputs.json','landing_mpc_records.json']]
    rows = []
    for case, start in selections:
        if not records[case,start]['accepted']: raise ValueError('Require accepted starting plan')
        path = a.run/'fixed_policy_traces'/f'case_{case:03d}.npz'
        sources.append(path)
        with np.load(path, allow_pickle=False) as z:
            public = unpack_public(z['packed_public'])
            actions = z['dispatched_actions']
        # Exact original final action -> joint target mapping, checked against
        # every saved controller record used by either replay.
        targets = interface.limits[:,0] + .5*(np.clip(actions,-1.,1.)+1.)*(
            interface.limits[:,1]-interface.limits[:,0])
        end = min(start+40,len(public)-1)
        target_error = 0.
        for tick in range(start,end):
            rec = records.get((case,tick))
            if rec is not None:
                target_error = max(target_error, float(np.max(abs(targets[tick]-
                    np.asarray(rec['final_dispatched_joint_target_rad'])))))
            if abs(float(public[tick,775])-.01*tick)>1e-6:
                raise ValueError('Unsupported first-episode clock')
        if target_error > 5e-6: raise ValueError('Executed target mapping mismatch')
        initial = model.state_from_public(public[start],default)
        initial.contacts = np.asarray(records[case,start]['support_contacts'],dtype=bool)
        row = dict(case=case,start_tick=start,available_steps=end-start,
                   target_mapping_error_max_rad=target_error,models={})
        checkpoints = {1,5,10,20,40,end-start}
        for mode in ['held_initial_contacts','oracle_recorded_force_contacts']:
            predicted = copy.deepcopy(initial)
            points = []
            maxima = {}
            failure = None
            for tick in range(start,end):
                if mode == 'oracle_recorded_force_contacts':
                    predicted.contacts = public[tick,57:61]>.5
                try:
                    out = predict_implicit_support(model,interface,predicted,targets[tick],substeps=16)
                    predicted = out['state']
                    for k,v in out['diagnostics'].items(): maxima[k] = max(maxima.get(k,0.),v)
                    if tick+1-start in checkpoints:
                        actual = model.state_from_public(public[tick+1],default)
                        points.append(dict(steps=tick+1-start,**errors(predicted,actual,model)))
                except (ValueError,np.linalg.LinAlgError) as exc:
                    failure = dict(tick=tick,message=str(exc))
                    break
            row['models'][mode] = dict(checkpoints=points,diagnostic_maxima=maxima,failure=failure)
        rows.append(row)
    sources += [Path(__file__), ROOT/'scripts/implicit_support_prediction_v1.py',
                ROOT/'scripts/loaded_pinocchio_model_v1.py', ROOT/'quadsmash/rl/episode_trace_v4.py']
    result = dict(scope=__doc__,run=str(a.run),native_new_execution=False,
        causal_deployment=False,rows=rows,
        source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(windows=len(rows),output=str(a.output))))


if __name__ == '__main__':
    main()
