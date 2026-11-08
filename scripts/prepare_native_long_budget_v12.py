"""Prepare full TRAIN128 native recovery with one full-authority command owner."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['parent-plan','output','native-output','fixture-run','library']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if a.output.exists() or a.native_output.exists():raise ValueError('Fresh experiment outputs required')
    plan=json.loads(a.parent_plan.read_text())
    for path,digest in plan['source_sha256'].items():
        if sha(path)!=digest:raise ValueError('Changed parent source '+path)
    old=next(Path(x) for x in plan['command'] if x.endswith('/evaluate_pure_control.py'))
    text=old.read_text()
    edits={
        'from scripts.native_pure_control_trajectory_recovery_v9 import NativePureControlTrajectoryRecoveryV9 as NativePureControlRecoveryV1':
        'from scripts.native_pure_control_long_trajectory_recovery_v11 import NativePureControlLongTrajectoryRecoveryV11 as NativePureControlRecoveryV1'}
    for before,after in edits.items():
        if text.count(before)!=1:raise ValueError('Ambiguous controller/observer wiring')
        text=text.replace(before,after)
    ast.parse(text);a.output.mkdir(parents=True)
    entry=a.output/'evaluate_pure_control.py';entry.write_text(text)
    command=plan['command'];old_config=Path(command[command.index('--pure-control-configuration')+1])
    configuration=json.loads(old_config.read_text())
    configuration.update(method='offline full18 trajectory with coupled predictive command selection and continuous support feedback',
        controller='pure_control_long_trajectory_recovery_v11',hip_mount_avoidance=True,model_reference_library=str(a.library.resolve()),software_arm_command_slew=None,
        original_hardware_unchanged=True,sequence_mpc=False,training=False)
    config=a.output/'controller.json';config.write_text(json.dumps(configuration,indent=2)+'\n')
    command=[str(entry.resolve()) if x==str(old) else x for x in command]
    for i,x in enumerate(command[:-1]):
        if x=='--output':command[i+1]=str(a.native_output.resolve())
        if x=='--pure-control-configuration':command[i+1]=str(config.resolve())
    limit_index=command.index('--max-seconds')+1
    if int(command[limit_index])!=4200:raise ValueError('Unexpected predecessor budget')
    command[limit_index]='5400'
    plan.update(command=command,declared_max_native_seconds=5400,scope=__doc__,output=str(a.native_output.resolve()),entry_sha256=sha(entry),
        parent_plan=str(a.parent_plan.resolve()),model_accepted=False,training=False,valid_expert_labels=False,
        check_command=['bash','scripts/python.sh','scripts/check_pure_control_long_trajectory_v11.py',
            '--entry',str(entry.resolve()),'--configuration',str(config.resolve()),
            '--run',str(a.fixture_run.resolve()),'--output',str((a.output/'entry_check.json').resolve())],
        post_command=['bash','scripts/python.sh','scripts/audit_native_pure_control_v2.py',
            '--run',str(a.native_output.resolve()),'--output',str((a.output/'native_audit.json').resolve())])
    paths=[Path(__file__),a.parent_plan,entry,config,ROOT/'configs/go2_piper_authored_foot_spheres_v1.json']
    paths += [ROOT/'scripts'/name for name in ['pure_control_court_recovery_v4.py',
        'native_pure_control_court_recovery_v4.py','recovery_surface_constraints_v1.py',
        'full_authority_position_interface_v1.py','authored_net_clearance_v1.py',
        'foot_pair_clearance_kinematics_v1.py','native_landing_contact_observer_v2.py',
        'check_pure_control_long_trajectory_v11.py','audit_native_pure_control_v2.py']]
    paths += [ROOT/'scripts'/name for name in ['pure_control_self_collision_recovery_v5.py',
        'native_pure_control_self_collision_recovery_v5.py','pure_control_interval_self_collision_v2.py',
        'hip_mount_planning_constraints_v1.py','hip_mount_collision_response_v1.py']]
    paths.append(ROOT/'configs/go2_piper_hip_mount_boundaries_v1.json')
    paths += [ROOT/'scripts'/name for name in ['pure_control_trajectory_recovery_v9.py',
        'native_pure_control_trajectory_recovery_v9.py','recovery_reference_library_v1.py',
        'build_recovery_reference_library_v1.py','predict_self_contact_spheres_v2.py',
        'foot_separation_path_constraint_v1.py','frictional_position_limited_step_v1.py']]
    paths += [ROOT/'scripts'/name for name in ['pure_control_long_trajectory_recovery_v11.py','native_pure_control_long_trajectory_recovery_v11.py','recovery_reference_library_v2.py','build_long_recovery_reference_v2.py']]
    paths += [ROOT/'scripts'/'run_native_pure_control_long_v2.py',ROOT/'scripts'/'run_serial_pure_control_long_v2.py']
    paths.append(a.library)
    # Freeze transitive local Python imports as well as direct controller files.
    # This adds integrity coverage without importing Isaac or executing code.
    pending=list(paths);seen=set()
    while pending:
        path=Path(pending.pop()).resolve()
        if path in seen:continue
        seen.add(path)
        if path.suffix!='.py' or not path.is_relative_to(ROOT):continue
        tree=ast.parse(path.read_text());package=list(path.relative_to(ROOT).with_suffix('').parts[:-1])
        for node in ast.walk(tree):
            modules=[]
            if isinstance(node,ast.Import):modules=[alias.name for alias in node.names]
            elif isinstance(node,ast.ImportFrom):
                if node.level:
                    base=package[:len(package)-node.level+1]
                    module='.'.join(base+([node.module] if node.module else []))
                else:module=node.module or ''
                modules=[module]+[module+'.'+alias.name for alias in node.names]
            for module in modules:
                if module.split('.')[0] not in ['scripts','quadsmash']:continue
                base=ROOT.joinpath(*module.split('.'))
                for dependency in [base.with_suffix('.py'),base/'__init__.py']:
                    if dependency.is_file():pending.append(dependency)
    for path in seen:plan['source_sha256'][str(path.resolve())]=sha(path)
    for name in ['frictional_speed_limited_step_v1.py','frictional_impulse_step_v1.py']:
        if str((ROOT/'scripts'/name).resolve()) not in plan['source_sha256']:
            raise ValueError('Missing coupled model dependency '+name)
    target=a.output/'plan.json';target.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(plan=str(target),native_execution=False)))


if __name__=='__main__':main()
