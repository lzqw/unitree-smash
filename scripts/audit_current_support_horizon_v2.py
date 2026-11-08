"""Validate a model horizon using commands already executed in native physics.

Future recorded commands are used ONLY for response validation, never as a
planner's initialization, objective or deployable control. No native reset.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pinocchio as pin

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.pure_control_reference_clock_v12 import PureControlReferenceClockV12
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.audit_actual_handoff_model_v1 import state_from_record
from scripts.audit_settled_stance_model_v2 import ForceRecorder
from scripts.predict_self_contact_spheres_v2 import predict_self_contact_spheres
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','configuration','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cases',type=int,nargs='+',required=True)
    a=p.parse_args();read=lambda path:json.loads(path.read_text())
    if a.output.exists() or read(a.run/'job.json')['status']!='SUCCEEDED':
        raise ValueError('Fresh output and completed native evidence required')
    cfg=read(a.configuration)
    c=PureControlReferenceClockV12(PositionPDInterface(**cfg['interface']),
        read(a.run/'native_command_inputs.json')['default_joint_position_rad'],cfg['model_reference_library'])
    with np.load(a.run/'pure_control_commands.npz') as z:arrays={k:z[k].copy() for k in z.files}
    with np.load(a.run/'pure_control_ground_forces.npz') as z:peaks=z['interval_peak_normal_n'].copy()
    nonfoot=[i for i,n in enumerate(read(a.run/'pure_control_ground_forces.json')['body_names']) if not n.endswith('_foot')]
    diagnostics=[json.loads(s) for s in (a.run/'pure_control_diagnostics.jsonl').read_text().splitlines()]
    out=[];started=time.monotonic()
    for case in a.cases:
        ids=np.flatnonzero(arrays['lane']==case)
        lookup={int(arrays['tick'][i]):int(i) for i in ids}
        captures=[t for t,i in lookup.items() if diagnostics[i]['diagnostic'].get('terminal_candidate_now')]
        if len(captures)!=1:raise ValueError('Require one actual stance capture')
        owner_start=min(lookup);hits=np.flatnonzero(np.any(peaks[owner_start:,case][:,nonfoot]>0.,axis=1))
        begin=captures[0];first=owner_start+int(hits[0]) if len(hits) else None
        end=min(begin+100,first if first is not None else 799)
        if end<=begin:raise ValueError('Need a clean interval after capture')
        i=lookup[begin];record={k:arrays[k][i] for k in ['q','v','previous','ground_force']}
        record['phase']=diagnostics[i]['phase'];state=state_from_record(c.model,record)
        rows=[];failure=None
        for tick in range(begin,end):
            i,j=lookup[tick],lookup[tick+1]
            if (diagnostics[i]['tick'],diagnostics[i]['lane'])!=(tick,case):raise ValueError('Unaligned evidence')
            target=arrays['final'][i];recorder=ForceRecorder(c.model)
            try:
                pred=predict_self_contact_spheres(c.model,c.interface,state,target,c.geometry.spheres,
                    substeps=160,geometry=c.path_geometry,collision_response=recorder)
            except (ValueError,np.linalg.LinAlgError) as error:
                failure=dict(tick=tick,reason='model_rejected',error=repr(error));break
            bad={n:v for n,v in pred['diagnostics'].items() if
                (n=='nonfoot_ground_violation_m' and v>1e-6) or
                (n in ['joint_position_violation_rad','joint_velocity_violation_rad_s','motor_torque_violation_nm'] and v>3e-4)}
            if bad:
                failure=dict(tick=tick,reason='model_path_violation',violations=bad,collision=pred['first_ground_violation']);break
            prior=state.contacts.copy();state=pred['state'];state.previous_command_target=target.copy()
            state.contacts=np.where(prior,recorder.last_force[:,2]>.5,recorder.last_force[:,2]>1.)
            state.measured_force_world=recorder.last_force.copy()
            q,v=arrays['q'][j],arrays['v'][j]
            rp,ra=state.root_rotation_world,pin.Quaternion(q[3:7]).matrix()
            difference=pin.difference(c.model.model,q,state.q)
            rows.append(dict(tick=tick,elapsed_s=(tick-begin+1)*.01,
                position_error_m=float(np.linalg.norm(state.q[:3]-q[:3])),
                attitude_error_rad=float(np.linalg.norm(difference[3:6])),
                joint_position_error_rad=float(np.max(abs(state.q[c.model.q_indices]-q[c.model.q_indices]))),
                linear_velocity_error_m_s=float(np.linalg.norm(rp@state.v[:3]-ra@v[:3])),
                angular_velocity_error_rad_s=float(np.linalg.norm(rp@state.v[3:6]-ra@v[3:6])),
                joint_velocity_error_rad_s=float(np.max(abs(state.v[c.model.v_indices]-v[c.model.v_indices]))),
                predicted_contacts=state.contacts.copy(),actual_next_contacts=diagnostics[j]['phase']['ground_contacts']))
        out.append(dict(case=case,start_tick=begin,required_steps=end-begin,steps=len(rows),
            actual_first_nonfoot_force_tick=first,failure=failure,trajectory=rows))
    files=[Path(__file__),a.configuration,Path(cfg['model_reference_library'])]+[a.run/n for n in
        ['job.json','native_command_inputs.json','pure_control_commands.npz','pure_control_diagnostics.jsonl',
         'pure_control_ground_forces.npz','pure_control_ground_forces.json']]
    files += [Path(m.__file__) for n,m in list(sys.modules.items()) if n.startswith('scripts.') and getattr(m,'__file__',None)]
    result=dict(results=out,wall_seconds=time.monotonic()-started,command_sequence_from_native=True,
        future_commands_for_diagnostic_only=True,valid_planning_input=False,native_success=False,training=False,
        substeps=160,source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix('.tmp')
    tmp.write_text(json.dumps(plain(result),indent=2,allow_nan=False)+'\n');tmp.replace(a.output)
    print(json.dumps(plain([dict(case=r['case'],steps=r['steps'],required_steps=r['required_steps'],failure=r['failure'],
        end_error=r['trajectory'][-1] if r['trajectory'] else None) for r in out])))


if __name__=='__main__':main()
