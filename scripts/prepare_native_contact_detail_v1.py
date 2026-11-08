"""Clone a pinned pure-control execution, changing only read-only observation."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['parent-plan','output','native-output']:
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if a.output.exists() or a.native_output.exists():raise ValueError('Fresh output required')
    plan=json.loads(a.parent_plan.read_text())
    for path,digest in plan['source_sha256'].items():
        if sha(path)!=digest:raise ValueError('Changed pinned source '+path)
    old=next(Path(x) for x in plan['command'] if x.endswith('/evaluate_pure_control.py'))
    before='from scripts.native_pure_control_observation_v1 import NativePureControlObservationV1'
    after='from scripts.native_landing_contact_observer_v2 import NativeLandingContactObserverV2 as NativePureControlObservationV1'
    text=old.read_text()
    if text.count(before)!=1:raise ValueError('Ambiguous observation edit')
    text=text.replace(before,after);ast.parse(text)
    a.output.mkdir(parents=True)
    entry=a.output/'evaluate_pure_control.py';entry.write_text(text)
    plan['command']=[str(entry.resolve()) if x==str(old) else x for x in plan['command']]
    for i,x in enumerate(plan['command'][:-1]):
        if x=='--output':plan['command'][i+1]=str(a.native_output.resolve())
    plan['output']=str(a.native_output.resolve());plan['entry_sha256']=sha(entry)
    for command,flag,path in [(plan['check_command'],'--entry',entry),
        (plan['check_command'],'--output',a.output/'entry_check.json'),
        (plan['post_command'],'--run',a.native_output),
        (plan['post_command'],'--output',a.output/'native_audit.json')]:
        command[command.index(flag)+1]=str(path.resolve())
    plan['post_command']=[x.replace('audit_native_pure_control_v1.py','audit_native_pure_control_v2.py')
                          for x in plan['post_command']]
    plan.update(scope=__doc__,parent_plan=str(a.parent_plan.resolve()),
        control_logic_changed=False,physics_parameters_changed=False,
        objective='Measure actual foot friction/contact response for the same pure-control commands')
    for path in [a.parent_plan,Path(__file__),entry,
        ROOT/'scripts/native_landing_contact_observer_v2.py',
        ROOT/'scripts/audit_native_pure_control_v2.py',
        ROOT/'tests/test_native_landing_contact_observer_v2.py']:
        plan['source_sha256'][str(path.resolve())]=sha(path)
    target=a.output/'plan.json';target.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(plan=str(target),native_execution=False)))


if __name__=='__main__':main()
