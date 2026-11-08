"""Prepare a separate native check of a still-rejected timed recovery proposal."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['parent-plan','generation','output','native-output']:
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or a.native_output.exists():raise ValueError('Fresh output paths required')
    read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    parent=read(a.parent_plan);old_bundle=Path(parent['bundle']);old=read(old_bundle/'manifest.json')
    generation=read(a.generation)
    rows=[r for r in generation['fine'] if (r['result']['failure'] or {}).get('reason')!='declared_compute_budget']
    chosen=min(rows,key=lambda r:r['result']['rank']);result=chosen['result']
    if (chosen['plan']['contract']!='timed_foot_space_recovery_v1' or result['accepted']
            or old['case']!=85 or result['control_tick']!=280):
        raise ValueError('Explicit still-rejected timed TRAIN85 proposal required')
    for key,field in [('initial_q','q'),('initial_v','v'),('initial_contacts','contacts'),('initial_previous_command','target')]:
        if result[key]!=old['query'][field]:raise ValueError('Measured initial state changed')
    pins=dict(parent['source_sha256'])
    for path,digest in pins.items():
        if sha(path)!=digest:raise ValueError('Parent evidence changed '+path)
    for path in [a.parent_plan,a.generation,Path(__file__).resolve(),
            ROOT/'scripts/native_timed_experimental_recovery_v1.py',ROOT/'scripts/timed_whole_body_recovery_v1.py']:
        pins[str(path.resolve())]=sha(path)
    a.output.mkdir(parents=True);bundle=a.output/'bundle';bundle.mkdir()
    archive=bundle/old['bundle_name'];shutil.copyfile(old_bundle/old['bundle_name'],archive)
    if sha(archive)!=old['bundle_sha256']:raise ValueError('Reference copy differs')
    m=copy.deepcopy(old);m.update(contract='timed_model_rejected_native_diagnostic_v1',
        proposal=chosen['plan'],model_rejection=result['failure'],model_completed_steps=result['completed_model_steps'],
        source_sha256=pins,model_accepted=False,valid_expert_labels=False,native_execution=False)
    (bundle/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    command=copy.deepcopy(parent['command'])
    old_entry=next(Path(v) for v in command if v.endswith('evaluate_experimental_recovery.py'))
    source=old_entry.read_text()
    before='from scripts.native_experimental_recovery_v2 import NativeExperimentalRecoveryV2 as NativeRecoverySequenceReplay'
    if source.count(before)!=1:raise ValueError('Unexpected evaluator template')
    source=source.replace(before,'from scripts.native_timed_experimental_recovery_v1 import NativeTimedExperimentalRecoveryV1 as NativeRecoverySequenceReplay')
    ast.parse(source);entry=a.output/'evaluate_timed_recovery.py';entry.write_text(source)
    command=[str(entry.resolve()) if v==str(old_entry) else v for v in command]
    for i,v in enumerate(command[:-1]):
        if v=='--output':command[i+1]=str(a.native_output.resolve())
        elif v=='--recovery-sequence-bundle':command[i+1]=str(bundle.resolve())
    # Retain the existing independent-process positive/negative query checks.
    check_template=ROOT/'scripts/check_native_experimental_recovery_v2.py'
    text=check_template.read_text().replace('from scripts.native_experimental_recovery_v2 import NativeExperimentalRecoveryV2',
        'from scripts.native_timed_experimental_recovery_v1 import NativeTimedExperimentalRecoveryV1')
    text=text.replace('NativeExperimentalRecoveryV2(', 'NativeTimedExperimentalRecoveryV1(')
    text=text.replace("ROOT=Path(__file__).resolve().parents[1]",'ROOT=Path('+repr(str(ROOT))+')')
    ast.parse(text);check=a.output/'check_fresh_process.py';check.write_text(text)
    for path in [entry,check,check_template,bundle/'manifest.json',archive]:pins[str(path.resolve())]=sha(path)
    plan=dict(scope=__doc__,output=str(a.native_output.resolve()),bundle=str(bundle.resolve()),
        command=command,entry_sha256=sha(entry),source_sha256=pins,model_accepted=False,
        automatic_promotion=False,valid_expert_labels=False,
        check_command=['bash','scripts/python.sh',str(check.resolve()),'--bundle',str(bundle.resolve()),
            '--output',str((a.output/'fresh_process_check').resolve())],
        post_command=['bash','scripts/python.sh','scripts/audit_native_experimental_recovery_v1.py',
            '--run',str(a.native_output.resolve()),'--bundle',str(bundle.resolve()),
            '--output',str((a.output/'native_audit.json').resolve())])
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(plan=str(a.output/'plan.json'),case=85,model_completed_steps=result['completed_model_steps'],
        model_rejection=result['failure'],native_execution=False)))


if __name__=='__main__':main()
