"""Prepare original full TRAIN128 V15 execution with archived references rechecked from current actual initial states."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['parent-plan','library','coverage','preservation','feedback','end-check','output','native-output']:
        p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();read=lambda f:json.loads(Path(f).read_text());sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    if a.output.exists() or a.native_output.exists():raise ValueError('Fresh native preparation required')
    plan=read(a.parent_plan)
    for f,h in plan['source_sha256'].items():
        if sha(f)!=h:raise ValueError('Changed verified baseline '+f)
    command=plan['command'];old=next(Path(x) for x in command if x.endswith('/evaluate_pure_control.py'))
    cfg=read(command[command.index('--pure-control-configuration')+1])
    cfg.update(model_reference_library=str(a.library.resolve()),
        method='verified V15 with unchanged original references plus full-model checked causal release references')
    a.output.mkdir(parents=True);entry=a.output/'evaluate_pure_control.py';entry.write_bytes(old.read_bytes())
    config=a.output/'controller.json';config.write_text(json.dumps(cfg,indent=2)+'\n')
    command=[str(entry.resolve()) if x==str(old) else x for x in command]
    command[command.index('--max-seconds')+1]='7200'
    for i,x in enumerate(command[:-1]):
        if x=='--output':command[i+1]=str(a.native_output.resolve())
        if x=='--pure-control-configuration':command[i+1]=str(config.resolve())
    check=['bash','scripts/python.sh','scripts/check_archived_library_native_entry_v2.py',
        '--parent-plan',str(a.parent_plan.resolve()),'--entry',str(entry.resolve()),
        '--configuration',str(config.resolve()),'--coverage',str(a.coverage.resolve()),
        '--preservation',str(a.preservation.resolve()),'--feedback',str(a.feedback.resolve()),
        '--end-check',str(a.end_check.resolve()),'--output',str((a.output/'entry_check.json').resolve())]
    subprocess.run([*check[:-1],str((a.output/'prepared_entry_check.json').resolve())],cwd=ROOT,check=True)
    receipt=read(a.output/'prepared_entry_check.json')
    plan['source_sha256'].update(receipt['source_sha256'])
    for f in [Path(__file__),a.output/'prepared_entry_check.json',ROOT/'scripts/run_native_pure_control_long_v3.py',ROOT/'scripts/run_serial_pure_control_current_v4.py']:
        plan['source_sha256'][str(f.resolve())]=sha(f)
    plan.update(command=command,scope=__doc__,declared_max_native_seconds=7200,
        output=str(a.native_output.resolve()),entry_sha256=sha(entry),
        parent_plan=str(a.parent_plan.resolve()),check_command=check,
        post_command=['bash','scripts/python.sh','scripts/audit_native_pure_control_v2.py',
            '--run',str(a.native_output.resolve()),'--output',str((a.output/'native_audit.json').resolve())],
        model_accepted=False,training=False,valid_expert_labels=False)
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(plan=str(a.output/'plan.json'),native_execution=False)))


if __name__=='__main__':main()
