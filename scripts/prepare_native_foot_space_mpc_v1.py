"""Version a native current-state planning entry without changing the task."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['source-plan','model-plan','model-search','output','bundle']:
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    out=a.output.resolve();bundle=a.bundle.resolve()
    if out.exists() or bundle.exists():raise ValueError('Fresh preparation and bundle required')
    original=json.loads(a.source_plan.read_text());cmd=original['command']
    source=Path(cmd[cmd.index('-u')+1]);old_bundle=Path(cmd[cmd.index('--recovery-sequence-bundle')+1])
    if sha(source)!=original['entry_sha256']:raise ValueError('Executed entry changed')
    for path,digest in original['source_sha256'].items():
        if sha(root/path)!=digest:raise ValueError('Changed original pinned dependency '+path)
    text=source.read_text()
    old='from scripts.native_foot_space_recovery_v1 import NativeFootSpaceRecoveryV1 as NativeRecoverySequenceReplay'
    if text.count(old)!=1:raise ValueError('Unexpected native controller import')
    text=text.replace(old,'from scripts.native_foot_space_mpc_v1 import NativeFootSpaceMPCV1 as NativeRecoverySequenceReplay')
    cleanup="    finally:\n        if 'head_observer' in locals():"
    if text.count(cleanup)!=1:raise ValueError('Unexpected native cleanup')
    text=text.replace(cleanup,"    finally:\n        if 'native_replay' in locals():\n            native_replay.close()\n        if 'head_observer' in locals():")
    ast.parse(text)
    audit_source=Path(original['post_commands'][-1][2]);audit=audit_source.read_text()
    old='from scripts.check_foot_space_dispatch_v1 import verify_feedback_dispatch'
    if audit.count(old)!=1:raise ValueError('Unexpected strict audit interface')
    audit=audit.replace(old,'from scripts.check_native_foot_space_mpc_v1 import verify_feedback_dispatch')
    audit=audit.replace('measured_state_feedback_dispatch=feedback_check,online_mpc=False',
        'measured_state_feedback_dispatch=feedback_check,online_mpc=True')
    ast.parse(audit);out.mkdir(parents=True);bundle.mkdir(parents=True)
    entry=out/'evaluate_foot_space_mpc.py';entry.write_text(text)
    audit_path=out/'audit_foot_space_mpc.py';audit_path.write_text(audit)
    manifest=json.loads((old_bundle/'manifest.json').read_text())
    shutil.copyfile(old_bundle/manifest['bundle_name'],bundle/manifest['bundle_name'])
    manifest['foot_space_current_state_planning']=dict(control_ticks=[400,550,700,750],
        max_seconds_per_call=600.,model_search=str(a.model_search.resolve()),source_plan=str(a.model_plan.resolve()))
    pins=dict(original['source_sha256'])
    for path in [Path(__file__),source,audit_source,a.source_plan,a.model_plan,a.model_search,entry,audit_path,
                 *[root/'scripts'/name for name in ['native_foot_space_mpc_v1.py',
                    'current_state_recovery_planner_v3.py','current_state_recovery_service_v3.py',
                    'check_native_foot_space_mpc_v1.py']]]:
        pins[str(path.resolve())]=sha(path)
    manifest['source_sha256'].update(pins)
    path=bundle/'manifest.json';path.write_text(json.dumps(manifest,indent=2)+'\n');pins[str(path)]=sha(path)
    result=dict(entry=str(entry),entry_sha256=sha(entry),audit=str(audit_path),bundle=str(bundle),
        source_sha256=pins,source_controller_run=original['output'],native_execution=False,
        physical_task_changed=False,strict_original_isolation_unchanged=True,valid_expert_labels=False,
        planning_ticks=[400,550,700,750],maximum_cpu_seconds_per_call=600.)
    (out/'preparation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}))


if __name__=='__main__':main()
