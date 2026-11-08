"""Prepare an original-prefix, state-matched full TRAIN128 replanning trial."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['parent-plan','replanning-library','output','native-output','fixture-run']:
        p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    if a.output.exists() or a.native_output.exists():raise ValueError('Fresh outputs required')
    plan=json.loads(a.parent_plan.read_text())
    for f,h in plan['source_sha256'].items():
        if sha(f)!=h:raise ValueError('Changed parent '+f)
    library=json.loads(a.replanning_library.read_text())
    if library['native_success'] or library['valid_expert_labels']:raise ValueError('No native success claimed before execution')
    for f,h in library['source_sha256'].items():
        if sha(f)!=h:raise ValueError('Changed planning source '+f)
    command=plan['command'];old=next(Path(x) for x in command if x.endswith('/evaluate_pure_control.py'))
    before='from scripts.native_pure_control_reference_clock_v12 import NativePureControlReferenceClockV12 as NativePureControlRecoveryV1'
    after='from scripts.native_pure_control_state_replanning_v16 import NativePureControlStateReplanningV16 as NativePureControlRecoveryV1'
    text=old.read_text()
    if text.count(before)!=1:raise ValueError('Preserved v12 prefix required')
    text=text.replace(before,after);ast.parse(text)
    cfg=json.loads(Path(command[command.index('--pure-control-configuration')+1]).read_text())
    cfg.update(controller='pure_control_state_replanning_v16',model_replanning_library=str(a.replanning_library.resolve()),
        method='unchanged v12 prefix with causal local-state selection of a computed recovery plan and full18 feedback')
    a.output.mkdir(parents=True);entry=a.output/'evaluate_pure_control.py';entry.write_text(text)
    config=a.output/'controller.json';config.write_text(json.dumps(cfg,indent=2)+'\n')
    command=[str(entry.resolve()) if x==str(old) else x for x in command]
    for i,x in enumerate(command[:-1]):
        if x=='--output':command[i+1]=str(a.native_output.resolve())
        if x=='--pure-control-configuration':command[i+1]=str(config.resolve())
    plan.update(command=command,scope=__doc__,output=str(a.native_output.resolve()),entry_sha256=sha(entry),
        parent_plan=str(a.parent_plan.resolve()),
        check_command=['bash','scripts/python.sh','scripts/check_native_state_replanning_v16.py',
            '--entry',str(entry.resolve()),'--configuration',str(config.resolve()),
            '--run',str(a.fixture_run.resolve()),'--output',str((a.output/'entry_check.json').resolve())],
        post_command=['bash','scripts/python.sh','scripts/audit_native_pure_control_v2.py',
            '--run',str(a.native_output.resolve()),'--output',str((a.output/'native_audit.json').resolve())])
    files=[Path(__file__),a.parent_plan,a.replanning_library,entry,config]
    files += [ROOT/'scripts'/n for n in ['pure_control_state_replanning_v16.py','native_pure_control_state_replanning_v16.py',
        'check_native_state_replanning_v16.py','check_state_replanning_v16.py','pure_control_stance_foot_reference_v15.py',
        'pure_control_settled_stance_v14.py','pure_control_terminal_stance_v13.py','audit_actual_handoff_model_v1.py',
        'audit_settled_stance_model_v2.py','build_actual_replanning_library_v1.py']]
    # Include every transitive local import from the new entries, independently
    # of which branches the checker happened to execute.
    pending=list(files);seen=set()
    while pending:
        f=Path(pending.pop()).resolve()
        if f in seen:continue
        seen.add(f)
        if f.suffix!='.py' or not f.is_relative_to(ROOT):continue
        tree=ast.parse(f.read_text());package=list(f.relative_to(ROOT).with_suffix('').parts[:-1])
        for node in ast.walk(tree):
            modules=[]
            if isinstance(node,ast.Import):modules=[x.name for x in node.names]
            elif isinstance(node,ast.ImportFrom):
                base=package[:len(package)-node.level+1] if node.level else []
                module='.'.join(base+([node.module] if node.module else []));modules=[module]+[module+'.'+x.name for x in node.names]
            for module in modules:
                if module.split('.')[0] not in ['scripts','quadsmash']:continue
                base=ROOT.joinpath(*module.split('.'))
                for dependency in [base.with_suffix('.py'),base/'__init__.py']:
                    if dependency.is_file():pending.append(dependency)
    for f in seen:plan['source_sha256'][str(f)]=sha(f)
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(plan=str(a.output/'plan.json'),native_execution=False)))


if __name__=='__main__':main()
