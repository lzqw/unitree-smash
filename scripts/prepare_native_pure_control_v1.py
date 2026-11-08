"""Prepare a versioned full TRAIN128 pure-control recovery execution trial."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['parent-plan','output','native-output','inputs']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    read=lambda p:json.loads(Path(p).read_text())
    if a.output.exists() or a.native_output.exists():raise ValueError('Fresh outputs required')
    parent=read(a.parent_plan)
    for path,h in parent['source_sha256'].items():
        if sha(path)!=h:raise ValueError('Changed parent source '+path)
    command=list(parent['command']);old=next(Path(x) for x in command if x.endswith('evaluate_joint_timed_family.py'))
    text=old.read_text()
    def change(before,after):
        nonlocal text
        if text.count(before)!=1:raise ValueError('Ambiguous entry edit '+before[:100])
        text=text.replace(before,after)
    change("parser.add_argument('--recovery-sequence-bundle', type=Path, required=True)\nparser.add_argument('--recovery-sequence-case', type=int, required=True)",
        "parser.add_argument('--pure-control-configuration', type=Path, required=True)")
    change("        recovery_sequence_bundle=str(args.recovery_sequence_bundle.resolve()),\n        recovery_sequence_case=args.recovery_sequence_case,\n        case_time_conditioned_diagnostic=True,",
        "        pure_control_configuration=str(args.pure_control_configuration.resolve()),\n        causal_release_full_control=True,case_time_conditioned_diagnostic=False,")
    begin=text.index('        from scripts.native_timed_experimental_recovery_v1 import')
    end=text.index('        sequence_student_commands=[];',begin)
    text=text[:begin]+'''        from scripts.native_pure_control_observation_v1 import NativePureControlObservationV1
        from scripts.native_pure_control_recovery_v1 import NativePureControlRecoveryV1
        pure_observer=NativePureControlObservationV1(env,args.output)
        native_replay=NativePureControlRecoveryV1(env,limits,default,
            args.pure_control_configuration,args.output,pure_observer)
'''+text[end:]
    begin=text.index('        from scripts.native_isolation_microtrace_v1 import')
    end=text.index('        finished = torch.zeros',begin)
    text=text[:begin]+text[end:]
    begin=text.index('                ground_support.update(tick)')
    end=text.index('                sequence_final_commands.append',begin)
    text=text[:begin]+'''                sequence_student_commands.append(action.cpu().numpy().copy())
                action=native_replay.action(tick,obs,action,finished)
'''+text[end:]
    begin=text.index("                if hasattr(controller, 'effort_override'):")
    end=text.index('                observation, _, done, _ = wrapped.step(action)',begin)
    text=text[:begin]+'''                # Recovery owns the total position command; no old torque overlay.
                env.robot.set_joint_effort_target(torch.zeros_like(action),joint_ids=env.joint_ids)
'''+text[end:]
    change('                observation, _, done, _ = wrapped.step(action)',
        '                observation, _, done, _ = wrapped.step(action)\n                native_replay.verify_applied(finished)')
    begin=text.index('        summary["recovery_sequence_replay"] = dict(')
    end=text.index("        if checkpoint.get('residual_training_metadata'",begin)
    text=text[:begin]+'''        summary['pure_control_recovery']=sequence_result
'''+text[end:]
    begin=text.index("        if 'ground_support' in locals():")
    end=text.index('        env._diagnostic_observer=None',begin)
    text=text[:begin]+'''        if 'native_replay' in locals():
            native_replay.save()
        if 'pure_observer' in locals():
            pure_observer.save()
            pure_observer.detach()
'''+text[end:]
    ast.parse(text)
    if 'native_replay.case' in text or 'native_replay.start' in text:raise ValueError('Fixed takeover survived')
    a.output.mkdir(parents=True)
    entry=a.output/'evaluate_pure_control.py';entry.write_text(text)
    # Copy only the physically pinned interface, not old policy/controller gains.
    from scripts.check_constrained_recovery_suffix_v1 import source_context
    data=read(a.inputs);source=read(read(data['configuration']['model_plan'])['source'])
    *_,context=source_context(source['configuration']['source'])
    config=a.output/'controller.json';config.write_text(json.dumps(dict(interface=context[0]['interface'],
        method='event-phase full18 predictive WBC',sequence_mpc=False,training=False),indent=2)+'\n')
    command=[str(entry.resolve()) if x==str(old) else x for x in command]
    for option in ['--recovery-sequence-case','--recovery-sequence-bundle']:
        i=command.index(option);del command[i:i+2]
    command+=['--pure-control-configuration',str(config.resolve())]
    for i,x in enumerate(command[:-1]):
        if x=='--output':command[i+1]=str(a.native_output.resolve())
        elif x=='--max-seconds':command[i+1]='4200'
    pins=dict(parent['source_sha256'])
    for path in [a.parent_plan,a.inputs,Path(__file__),entry,config]+[
        ROOT/'scripts'/n for n in ['pure_control_interval_qp_v1.py','pure_control_recovery_v1.py',
        'native_pure_control_recovery_v1.py','native_pure_control_observation_v1.py','event_recovery_manager_v1.py',
        'check_pure_control_entry_v1.py','audit_native_pure_control_v1.py','run_native_pure_control_v1.py']]:
        pins[str(path.resolve())]=sha(path)
    plan=dict(scope=__doc__,source_sha256=pins,entry_sha256=sha(entry),output=str(a.native_output.resolve()),
        command=command,model_accepted=False,training=False,valid_expert_labels=False,
        check_command=['bash','scripts/python.sh','scripts/check_pure_control_entry_v1.py',
            '--entry',str(entry.resolve()),'--configuration',str(config.resolve()),'--inputs',str(a.inputs.resolve()),
            '--output',str((a.output/'entry_check.json').resolve())],
        post_command=['bash','scripts/python.sh','scripts/audit_native_pure_control_v1.py',
            '--run',str(a.native_output.resolve()),'--output',str((a.output/'native_audit.json').resolve())])
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(plan=str(a.output/'plan.json'),native_execution=False)))


if __name__=='__main__':main()
