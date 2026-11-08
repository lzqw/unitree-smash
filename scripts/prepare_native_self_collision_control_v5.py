"""Prepare full TRAIN128 native recovery with one full-authority command owner."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['parent-plan','output','native-output','fixture-run']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if a.output.exists() or a.native_output.exists():raise ValueError('Fresh experiment outputs required')
    plan=json.loads(a.parent_plan.read_text())
    for path,digest in plan['source_sha256'].items():
        if sha(path)!=digest:raise ValueError('Changed parent source '+path)
    old=next(Path(x) for x in plan['command'] if x.endswith('/evaluate_pure_control.py'))
    text=old.read_text()
    edits={
        'from scripts.native_pure_control_court_recovery_v4 import NativePureControlCourtRecoveryV4 as NativePureControlRecoveryV1':
        'from scripts.native_pure_control_self_collision_recovery_v5 import NativePureControlSelfCollisionRecoveryV5 as NativePureControlRecoveryV1'}
    for before,after in edits.items():
        if text.count(before)!=1:raise ValueError('Ambiguous controller/observer wiring')
        text=text.replace(before,after)
    ast.parse(text);a.output.mkdir(parents=True)
    entry=a.output/'evaluate_pure_control.py';entry.write_text(text)
    command=plan['command'];old_config=Path(command[command.index('--pure-control-configuration')+1])
    configuration=json.loads(old_config.read_text())
    configuration.update(method='event-phase full18 WBC with ground/net/foot-pair and hip-mount avoidance',
        controller='pure_control_self_collision_recovery_v5',hip_mount_avoidance=True,software_arm_command_slew=None,
        original_hardware_unchanged=True,sequence_mpc=False,training=False)
    config=a.output/'controller.json';config.write_text(json.dumps(configuration,indent=2)+'\n')
    command=[str(entry.resolve()) if x==str(old) else x for x in command]
    for i,x in enumerate(command[:-1]):
        if x=='--output':command[i+1]=str(a.native_output.resolve())
        if x=='--pure-control-configuration':command[i+1]=str(config.resolve())
    plan.update(command=command,scope=__doc__,output=str(a.native_output.resolve()),entry_sha256=sha(entry),
        parent_plan=str(a.parent_plan.resolve()),model_accepted=False,training=False,valid_expert_labels=False,
        check_command=['bash','scripts/python.sh','scripts/check_pure_control_self_collision_v5.py',
            '--entry',str(entry.resolve()),'--configuration',str(config.resolve()),
            '--run',str(a.fixture_run.resolve()),'--output',str((a.output/'entry_check.json').resolve())],
        post_command=['bash','scripts/python.sh','scripts/audit_native_pure_control_v2.py',
            '--run',str(a.native_output.resolve()),'--output',str((a.output/'native_audit.json').resolve())])
    paths=[Path(__file__),a.parent_plan,entry,config,ROOT/'configs/go2_piper_authored_foot_spheres_v1.json']
    paths += [ROOT/'scripts'/name for name in ['pure_control_court_recovery_v4.py',
        'native_pure_control_court_recovery_v4.py','recovery_surface_constraints_v1.py',
        'full_authority_position_interface_v1.py','authored_net_clearance_v1.py',
        'foot_pair_clearance_kinematics_v1.py','native_landing_contact_observer_v2.py',
        'check_pure_control_self_collision_v5.py','audit_native_pure_control_v2.py']]
    paths += [ROOT/'scripts'/name for name in ['pure_control_self_collision_recovery_v5.py',
        'native_pure_control_self_collision_recovery_v5.py','pure_control_interval_self_collision_v2.py',
        'hip_mount_planning_constraints_v1.py','hip_mount_collision_response_v1.py']]
    paths.append(ROOT/'configs/go2_piper_hip_mount_boundaries_v1.json')
    for path in paths:plan['source_sha256'][str(path.resolve())]=sha(path)
    target=a.output/'plan.json';target.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(plan=str(target),native_execution=False)))


if __name__=='__main__':main()
