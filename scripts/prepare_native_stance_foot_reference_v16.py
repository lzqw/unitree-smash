"""Pin a full TRAIN128 foothold-objective control trial with unchanged inputs."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['parent-plan','output','native-output','fixture-run']:
        p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    if a.output.exists() or a.native_output.exists():raise ValueError('Fresh outputs required')
    plan=json.loads(a.parent_plan.read_text())
    for f,h in plan['source_sha256'].items():
        if sha(f)!=h:raise ValueError('Changed parent '+f)
    command=plan['command'];old=next(Path(x) for x in command if x.endswith('/evaluate_pure_control.py'))
    before='from scripts.native_pure_control_settled_stance_v14 import NativePureControlSettledStanceV14 as NativePureControlRecoveryV1'
    after='from scripts.native_pure_control_stance_foot_reference_v15 import NativePureControlStanceFootReferenceV15 as NativePureControlRecoveryV1'
    text=old.read_text()
    if text.count(before)!=1:raise ValueError('Unexpected predecessor entry')
    text=text.replace(before,after);ast.parse(text)
    cfg=json.loads(Path(command[command.index('--pure-control-configuration')+1]).read_text())
    cfg.update(controller='pure_control_stance_foot_reference_v15',
        method='state-qualified terminal body and foot soft objectives with unchanged observed-contact constraints')
    a.output.mkdir(parents=True);entry=a.output/'evaluate_pure_control.py';entry.write_text(text)
    config=a.output/'controller.json';config.write_text(json.dumps(cfg,indent=2)+'\n')
    command=[str(entry.resolve()) if x==str(old) else x for x in command]
    for i,x in enumerate(command[:-1]):
        if x=='--output':command[i+1]=str(a.native_output.resolve())
        if x=='--pure-control-configuration':command[i+1]=str(config.resolve())
    if int(command[command.index('--max-seconds')+1])!=5400:raise ValueError('Changed per-job bound')
    plan.update(command=command,scope=__doc__,output=str(a.native_output.resolve()),
        entry_sha256=sha(entry),parent_plan=str(a.parent_plan.resolve()),
        check_command=['bash','scripts/python.sh','scripts/check_native_stance_foot_reference_v15.py',
            '--entry',str(entry.resolve()),'--configuration',str(config.resolve()),
            '--run',str(a.fixture_run.resolve()),'--output',str((a.output/'entry_check.json').resolve())],
        post_command=['bash','scripts/python.sh','scripts/audit_native_pure_control_v2.py',
            '--run',str(a.native_output.resolve()),'--output',str((a.output/'native_audit.json').resolve())])
    files=[Path(__file__),a.parent_plan,entry,config]
    files += [ROOT/'scripts'/n for n in ['pure_control_stance_foot_reference_v15.py',
        'native_pure_control_stance_foot_reference_v15.py','check_native_stance_foot_reference_v15.py']]
    for f in files:plan['source_sha256'][str(f.resolve())]=sha(f)
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(plan=str(a.output/'plan.json'),native_execution=False)))


if __name__=='__main__':main()
