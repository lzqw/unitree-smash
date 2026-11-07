"""Continue a verified nominal recovery to the original episode boundary.

Every block is replanned from the previous predicted endpoint, with original
command history and physics. This is offline nominal planning, not evidence
of native feedback or a successful task episode.
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
import scripts.landing_sequential_recovery_search_v1 as search
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.terminal_recovery_constraints_v1 import make_proposal
from scripts.audit_ramped_gravity_continuation_v1 import future_equilibrium_reference,rate_admitted_seed
from scripts.audit_microstep_recovery_v1 import characterize
from scripts.audit_joint_sequence_proposal_v1 import plain


def kinematic_window(model,states,protocol,dt):
    """Same kinematic thresholds, without claiming native force/contact validity."""
    run=0.;longest=0.;count=0
    for s in states:
        rpy=pin.rpy.matrixToRpy(s.root_rotation_world)
        linear=s.root_rotation_world@s.v[:3]
        ok=bool(s.contacts.any() and max(abs(rpy[:2]))<=np.deg2rad(protocol['maximum_abs_roll_pitch_deg'])
            and np.linalg.norm(linear[:2])<=protocol['maximum_horizontal_root_speed_m_s']
            and np.linalg.norm(s.v[3:6])<=protocol['maximum_root_angular_speed_rad_s']
            and max(abs(s.joint_velocity))<=protocol['maximum_abs_joint_velocity_rad_s'])
        run=run+dt if ok else 0.;longest=max(longest,run);count+=int(ok)
    return dict(longest_model_kinematic_window_s=longest,qualifying_control_ticks=count,
        required_window_s=protocol['continuous_stable_window_s'],native_force_check=False,
        contact_flight_association_check=False,formal_recovery_success=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['bundle','geometry','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--case',type=int,default=3);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve previous extension')
    m=json.loads((a.bundle/'manifest.json').read_text());archive=a.bundle/m['bundle_name']
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=m['bundle_sha256']:raise ValueError('Changed prefix')
    for name,digest in m['source_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed prefix source '+name)
    row=next(r for r in m['rows'] if r['case']==a.case);cfg=m['configuration'];c=build_controller(cfg)
    c.sequence_geometry=AuthoredGroundClearance(a.geometry,c.qp.model,margin_m=0.)
    key=f'case_{a.case:03d}';limits=np.asarray(cfg['interface']['limits'])
    with np.load(archive,allow_pickle=False) as z:
        state=c.qp.model.state_from_public(z[key+'_query_public'],np.asarray(cfg['posture_target']))
        state.previous_command_target=limits[:,0]+.5*(z[key+'_previous_action']+1.)*np.diff(limits).ravel()
        reference=limits[:,0]+.5*(z[key+'_query_student_action']+1.)*np.diff(limits).ravel()
        commands=z[key+'_commands_rad'].copy();expected_q=z[key+'_predicted_q'].copy();expected_v=z[key+'_predicted_v'].copy()
    if len(commands)%40:raise ValueError('Require complete independently verified prefix blocks')
    states=[copy.deepcopy(state)];blocks=[]
    for offset in range(0,len(commands),40):
        check,_=characterize(c,state,commands[offset:offset+40],reference)
        replay=search.rollout_for_search(c,state,commands[offset:offset+40],reference)
        if not check['accepted']:raise ValueError('Prefix no longer passes')
        for i,n in enumerate(replay['profile'],offset+1):
            actual=n['following']
            if np.max(abs(pin.difference(c.qp.model.model,expected_q[i],actual.q)))>1e-8 or np.max(abs(expected_v[i]-actual.v))>1e-8:
                raise ValueError('Continuous prefix differs from independent export')
            states.append(copy.deepcopy(actual))
        state=copy.deepcopy(states[-1]);reference=state.previous_command_target.copy()
    initial_prefix_steps=len(commands);query_tick=row['query_tick'];needed_steps=800-query_tick
    if not 0<needed_steps<=800:raise ValueError('Original 8 second episode required')
    old_propose=search.propose_sequence;old_rollout=search.rollout_for_search
    def weighted_rollout(*args):
        result=old_rollout(*args);H=len(result['profile'])
        result['cost']+=sum((10. if i==H-1 else 1./H)*np.sum(n['following'].joint_velocity**2) for i,n in enumerate(result['profile']))
        return result
    search.propose_sequence=make_proposal(2.);search.rollout_for_search=weighted_rollout
    a.output.parent.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    try:
        while len(commands)<needed_steps and len(blocks)<20:
            reference=state.previous_command_target.copy();proposal=future_equilibrium_reference(c,state)
            block=dict(start_control_tick=query_tick+len(commands),proposal=proposal,accepted=False)
            if not proposal['solved']:blocks.append(block);break
            seed=rate_admitted_seed(c,state,proposal['target'],40)
            before,_=characterize(c,state,seed,reference);task=None;repair=None;chosen=seed
            if not before['accepted'] or not before['joint_settling_diagnostic']:
                task=search.sequential_recovery_search(c,state,reference,seed,iterations=12,time_limit_s=300.)
                repair=search.restore_feasibility(c,state,reference,task['final_search_commands'],iterations=12)
                chosen=repair['commands'] if repair['accepted'] else task['commands']
                if chosen is None:chosen=repair['final_search_commands']
            after,_=characterize(c,state,chosen,reference)
            block.update(before=before,after=after,task=task,repair=repair,
                accepted=bool(after['accepted'] and after['joint_settling_diagnostic']))
            blocks.append(block)
            print(json.dumps(dict(start_tick=block['start_control_tick'],accepted=block['accepted'],
                terminal=after['terminal_risk'],leg_speed=after['maximum_leg_speed_rad_s'])),flush=True)
            a.output.with_suffix('.partial.json').write_text(json.dumps(plain(blocks),indent=2)+'\n')
            if not block['accepted']:break
            replay=old_rollout(c,state,chosen,reference)
            states.extend(copy.deepcopy(n['following']) for n in replay['profile'])
            commands=np.concatenate([commands,chosen]);state=copy.deepcopy(states[-1])
    finally:search.propose_sequence=old_propose;search.rollout_for_search=old_rollout
    full=len(commands)>=needed_steps;protocol_path=ROOT/'configs/evaluation_protocol_v2.json'
    protocol=json.loads(protocol_path.read_text())['recovery'];length=min(len(commands),needed_steps)
    arrays=a.output.with_suffix('.npz');np.savez_compressed(arrays,
        commands_rad=commands[:length],predicted_q=np.stack([s.q for s in states[:length+1]]),
        predicted_v=np.stack([s.v for s in states[:length+1]]),predicted_contacts=np.stack([s.contacts for s in states[:length+1]]))
    files=[Path(__file__),a.bundle/'manifest.json',archive,a.geometry,protocol_path]
    files += [Path(m.__file__) for n,m in sys.modules.items() if n.startswith('scripts.') and getattr(m,'__file__',None)]
    result=dict(scope=__doc__,case=a.case,query_tick=query_tick,initial_prefix_steps=initial_prefix_steps,
        native_execution=False,valid_expert_labels=False,model_plan_reaches_original_end=full,
        exported_steps=length,checked_steps=len(commands),checked_end_time_s=.01*(query_tick+len(commands)),
        formal_recovery_kinematics=kinematic_window(c.qp.model,states[1:length+1],protocol,.01),
        blocks=blocks,wall_seconds=time.perf_counter()-started,arrays=str(arrays),
        arrays_sha256=hashlib.sha256(arrays.read_bytes()).hexdigest(),
        source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    a.output.write_text(json.dumps(plain(result),indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
