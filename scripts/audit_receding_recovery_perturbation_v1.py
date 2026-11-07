"""Compare fixed commands and observed-state replanning in the same model.

Artificial velocity perturbations are diagnostics, not formal task trials.
The plant shares the predictor: this cannot establish model accuracy or native
robustness. Every branch starts from the same perturbed nominal state.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pinocchio as pin
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.receding_recovery_sequence_v1 import RecedingRecoverySequenceV1,shifted_seed
from scripts.landing_sequential_recovery_search_v1 import rollout_for_search
from scripts.extend_recovery_to_episode_v1 import kinematic_window
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['bundle','geometry','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--start-tick',type=int,default=660)
    p.add_argument('--horizontal-kick',type=float,default=.04)
    p.add_argument('--pitch-rate-kick',type=float,default=.05)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve diagnostic evidence')
    m=json.loads((a.bundle/'manifest.json').read_text());archive=a.bundle/m['bundle_name']
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=m['bundle_sha256']:raise ValueError('Changed nominal bundle')
    for name,digest in m['source_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed nominal source '+name)
    row=m['rows'][0];key=f"case_{row['case']:03d}";offset=a.start_tick-row['query_tick']
    if not 1<=offset<row['horizon_steps'] or not np.isfinite([a.horizontal_kick,a.pitch_rate_kick]).all():raise ValueError('Invalid diagnostic query')
    with np.load(archive,allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    c=build_controller(m['configuration']);c.sequence_geometry=AuthoredGroundClearance(a.geometry,c.qp.model,margin_m=0.)
    state=c.qp.model.state_from_public(data[key+'_query_public'],np.asarray(m['configuration']['posture_target']))
    state.q=data[key+'_predicted_q'][offset].copy();state.v=data[key+'_predicted_v'][offset].copy()
    state.contacts=data[key+'_predicted_contacts'][offset].copy()
    state.root_position_world=state.q[:3].copy();state.root_rotation_world=pin.Quaternion(state.q[3:7]).matrix()
    state.v[:3]+=state.root_rotation_world.T@np.array([a.horizontal_kick,0.,0.])
    state.v[3:6]+=state.root_rotation_world.T@np.array([0.,a.pitch_rate_kick,0.])
    state.joint_position=state.q[c.qp.model.q_indices].copy();state.joint_velocity=state.v[c.qp.model.v_indices].copy()
    state.previous_command_target=data[key+'_commands_rad'][offset-1].copy()
    original=data[key+'_commands_rad'][offset:];protocol_path=ROOT/'configs/evaluation_protocol_v2.json'
    protocol=json.loads(protocol_path.read_text())['recovery'];branches=[];started=time.perf_counter()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    for mode in ['fixed_plan','receding_observed_state']:
        current=copy.deepcopy(state);states=[];records=[];solver=RecedingRecoverySequenceV1(c)
        warm=shifted_seed(original,0,40,current.previous_command_target,c.qp.interface,.01)
        for step in range(0,len(original),10):
            count=min(10,len(original)-step);reference=current.previous_command_target.copy();decision=None
            if mode=='fixed_plan':commands=shifted_seed(original,step,40,reference,c.qp.interface,.01)
            else:
                decision=solver.solve(current,reference,warm)
                if not decision['accepted']:
                    records.append(dict(tick=a.start_tick+step,decision=decision,executed_steps=0));break
                commands=decision['commands']
                if not np.array_equal(decision['initial_v'],current.v):raise ValueError('Planner ignored observed velocity')
            rollout=rollout_for_search(c,current,commands,reference)
            nodes=rollout['profile'][:count];states.extend(copy.deepcopy(n['following']) for n in nodes)
            record=dict(tick=a.start_tick+step,decision=decision,executed_steps=count,
                physical_checks=rollout['checks'][:count],command_difference_from_fixed_rad=
                float(np.max(abs(commands[:count]-original[step:step+count]))),
                terminal_risk=c.risk(states[-1]))
            records.append(record);current=copy.deepcopy(states[-1])
            warm=shifted_seed(commands,count,40,current.previous_command_target,c.qp.interface,.01)
            print(json.dumps(dict(mode=mode,tick=record['tick'],tilt=record['terminal_risk']['tilt_rad'],
                command_change=record['command_difference_from_fixed_rad'])),flush=True)
            a.output.with_suffix('.partial.json').write_text(json.dumps(plain(dict(mode=mode,records=records)),indent=2)+'\n')
        max_violation=max((max(ch['physical'].values()) for r in records for ch in r.get('physical_checks',[])),default=None)
        branches.append(dict(mode=mode,executed_steps=len(states),reached_episode_end=len(states)==len(original),
            records=records,maximum_executed_physical_violation=max_violation,
            kinematics=kinematic_window(c.qp.model,states,protocol,.01),
            terminal_risk=c.risk(current),native_execution=False,valid_expert_labels=False))
    files=[Path(__file__),a.bundle/'manifest.json',archive,a.geometry,protocol_path]
    files += [Path(module.__file__) for name,module in sys.modules.items() if name.startswith('scripts.') and getattr(module,'__file__',None)]
    report=dict(scope=__doc__,case=row['case'],start_tick=a.start_tick,
        artificial_world_velocity_kick_mps=[a.horizontal_kick,0,0],artificial_world_angular_kick_rad_s=[0,a.pitch_rate_kick,0],
        same_predictor_and_plant=True,native_execution=False,formal_task_evaluation=False,branches=branches,
        wall_seconds=time.perf_counter()-started,source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    a.output.write_text(json.dumps(plain(report),indent=2,allow_nan=False)+'\n')
    print(json.dumps({b['mode']:dict(steps=b['executed_steps'],maximum_violation=b['maximum_executed_physical_violation'],
        model_window=b['kinematics']['longest_model_kinematic_window_s']) for b in branches}),flush=True)


if __name__=='__main__':main()
