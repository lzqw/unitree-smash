"""Prepare a phase-adaptive library-independent MPC trial."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--native-output', type=Path, required=True)
    a = p.parse_args(); sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    if a.output.exists() or a.native_output.exists(): raise ValueError('Fresh outputs required')
    parent=json.loads(a.parent_plan.read_text())
    for f,h in parent['source_sha256'].items():
        if sha(f)!=h: raise ValueError('Changed parent source '+f)
    command=parent['command'][:]; old=next(Path(x) for x in command if x.endswith('/evaluate_pure_control.py'))
    before=('from scripts.native_pure_control_stance_foot_reference_v15 import '
            'NativePureControlStanceFootReferenceV15 as NativePureControlRecoveryV1')
    after=('from scripts.native_pure_control_state_driven_adaptive_mpc_v1 import '
           'NativePureControlStateDrivenAdaptiveMPCV1 as NativePureControlRecoveryV1')
    text=old.read_text()
    if text.count(before)!=1: raise ValueError('Unexpected predecessor entry')
    text=text.replace(before,after); ast.parse(text)
    cfg_path=Path(command[command.index('--pure-control-configuration')+1]); cfg=json.loads(cfg_path.read_text())
    cfg.update(controller='pure_control_state_driven_adaptive_mpc_v1',
               method='phase-adaptive state-driven rolling MPC without reference library',
               reference_library_required=False,train_case_required=False,sequence_mpc=True,
               fallback_certified=False)
    a.output.mkdir(parents=True); entry=a.output/'evaluate_pure_control.py'; entry.write_text(text)
    config=a.output/'controller.json'; config.write_text(json.dumps(cfg,indent=2)+'\n')
    command=[str(entry.resolve()) if x==str(old) else x for x in command]
    for i,x in enumerate(command[:-1]):
        if x=='--output': command[i+1]=str(a.native_output.resolve())
        if x=='--pure-control-configuration': command[i+1]=str(config.resolve())
    if int(command[command.index('--max-seconds')+1])!=7200: raise ValueError('Changed per-job bound')
    files=[Path(__file__),a.parent_plan,entry,config,
           Path(__file__).resolve().parents[1]/'scripts'/'pure_control_state_driven_adaptive_mpc_v1.py',
           Path(__file__).resolve().parents[1]/'scripts'/'native_pure_control_state_driven_adaptive_mpc_v1.py']
    pins=dict(parent['source_sha256']); pins.update({str(f.resolve()):sha(f) for f in files})
    plan=dict(parent,command=command,output=str(a.native_output.resolve()),scope=__doc__,
              entry_sha256=sha(entry),source_sha256=pins,reference_library_required=False,
              train_case_required=False,native_execution=False,
              check_command=['bash','scripts/python.sh','scripts/check_native_state_driven_adaptive_mpc_v1.py',
                             '--entry',str(entry.resolve()),'--configuration',str(config.resolve()),
                             '--output',str((a.output/'entry_check.json').resolve())])
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(plan=str(a.output/'plan.json'),native_execution=False)))


if __name__=='__main__': main()
