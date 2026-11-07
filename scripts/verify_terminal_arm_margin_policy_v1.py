"""Save a complete corrected control trajectory using 160 model substeps per tick.

The recorded native query is only the model initial condition. Native execution
still must start from the ground and pass its own full-episode audit.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import stress_terminal_support_commands_v1 as stress
from scripts.search_terminal_prefix_arm_margin_v1 import ArmPrefixCandidate
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['bundle','warm-plan','validation','sphere-audit','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve prior validation')
    stress.initialize(args.bundle,args.warm_plan,args.validation,args.sphere_audit)
    controller,base,initial,record,spheres,geometry,protocol=stress.CONTEXT
    policy=ArmPrefixCandidate(base,controller.qp.interface,-.006)
    files=[Path(__file__),args.bundle/'manifest.json',args.warm_plan,args.validation,args.sphere_audit,
           ROOT/'configs/evaluation_protocol_v2.json',controller.qp.model.path]
    files += [Path(value.__file__) for key,value in list(sys.modules.items())
              if key.startswith('scripts.') and getattr(value,'__file__',None)]
    digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    hashes={str(path.resolve()):digest(path) for path in files}
    state=copy.deepcopy(initial);states=[];commands=[];records=[];failure=None;start=time.monotonic()
    for offset in range(340):
        target,mode=policy.action(offset,state,state.previous_command_target)
        result=stress.predict_frictional_spheres(controller.qp.model,controller.qp.interface,state,
                                                target,spheres,geometry=geometry,substeps=160)
        diagnostics=result['diagnostics']
        bad={key:value for key,value in diagnostics.items() if
             (key=='nonfoot_ground_violation_m' and value>1e-6) or
             (key in ['joint_position_violation_rad','joint_velocity_violation_rad_s','motor_torque_violation_nm']
              and value>3e-4)}
        records.append(dict(offset=offset,mode=mode,diagnostics=diagnostics))
        if bad:failure=dict(offset=offset,violations=bad);break
        state=result['state'];state.previous_command_target=target.copy()
        states.append(copy.deepcopy(state));commands.append(target.copy())
    window=stress.kinematic_window(controller.qp.model,states,protocol,.01)
    for path,expected in hashes.items():
        if digest(path)!=expected:raise ValueError('Changed input: '+path)
    output=dict(scope=__doc__,case=record['case'],start_tick=460,needed_steps=340,
        executed_model_steps=len(states),model_candidate_accepted=failure is None and len(states)==340
        and window['longest_model_kinematic_window_s']>=.8,failure=failure,model_window=window,
        base_validation_path=str(args.validation.resolve()),base_validation_sha256=digest(args.validation),
        initial_q=initial.q,initial_v=initial.v,initial_contacts=initial.contacts,
        actual_previous_command=initial.previous_command_target,selected_tail=record['selected_tail'],
        arm_correction=dict(joint_index=13,amplitude_rad=-.006,start=15,end=45,profile='half_sine'),
        integration_substeps=160,states=[dict(q=s.q,v=s.v,contacts=s.contacts) for s in states],
        commands=commands,records=records,source_sha256=hashes,wall_seconds=time.monotonic()-start,
        measured_native_initial_state=True,native_continuation_executed=False,valid_expert_labels=False)
    temporary=args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(plain(output),indent=2,allow_nan=False)+'\n');temporary.replace(args.output)
    print(json.dumps({key:output[key] for key in ['model_candidate_accepted','executed_model_steps','failure','model_window']}))


if __name__=='__main__':main()
