"""Prepare bounded native current-state planning without changing the old run."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_native_early_support_v1 import prepare


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['base-bundle','interface-check','output','bundle-output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if a.output.exists() or a.bundle_output.exists():raise ValueError('Fresh preparation required')
    check=json.loads(a.interface_check.read_text())
    if (check['native_execution'] or not check.get('altered_query_rejected') or
            not check['service_closed'] or check['control_ticks']!=800 or
            check['dispatch_audit']['maximum_normalized_action_error']>1e-6):
        raise ValueError('Executed planning/dispatch interface check required')
    m=json.loads((a.base_bundle/'manifest.json').read_text())
    if sha(a.base_bundle/m['bundle_name'])!=m['bundle_sha256']:raise ValueError('Base bundle changed')
    for pins in [m['source_sha256'],check['source_sha256']]:
        for path,h in pins.items():
            if sha(path)!=h:raise ValueError('Checked source changed '+path)
    m['current_state_planning']=dict(control_ticks=[400,550,700,750],max_seconds_per_call=600.,
        workers=4,physics_waits_at_control_boundary=True,abort_if_no_verified_plan=True,
        case_time_conditioned_diagnostic=True,deployable_risk_trigger=False)
    m['scope']='Preserved native prefix, admitted offline suffix, then actual-state full-remainder planning.'
    m['source_sha256'].update(check['source_sha256'])
    paths=[a.interface_check.resolve(),a.base_bundle.resolve()/'manifest.json',Path(__file__).resolve(),
        ROOT/'scripts/native_current_state_recovery_v1.py',ROOT/'scripts/current_state_recovery_service_v1.py',
        ROOT/'scripts/current_state_recovery_planner_v1.py',ROOT/'scripts/audit_native_current_state_dispatch_v1.py']
    m['source_sha256'].update({str(path):sha(path) for path in paths})
    a.bundle_output.mkdir(parents=True)
    shutil.copyfile(a.base_bundle/m['bundle_name'],a.bundle_output/m['bundle_name'])
    (a.bundle_output/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    result=prepare(ROOT,a.output)
    def replace(text,before,after):
        if text.count(before)!=1:raise ValueError('Preparation template changed '+before)
        return text.replace(before,after)
    entry=Path(result['entry']);text=replace(entry.read_text(),
        'from scripts.native_early_support_tail_v1 import NativeEarlySupportTailV1 as NativeRecoverySequenceReplay',
        'from scripts.native_current_state_recovery_v1 import NativeCurrentStateRecoveryV1 as NativeRecoverySequenceReplay')
    ast.parse(text);entry.write_text(text)
    audit=Path(result['audit']);text=replace(audit.read_text(),
        'from scripts.check_early_support_dispatch_v1 import verify_feedback_dispatch',
        'from scripts.audit_native_current_state_dispatch_v1 import verify_feedback_dispatch')
    text=replace(text,"files += [Path(__file__),ROOT/'scripts/check_early_support_dispatch_v1.py',",
        "files += [Path(__file__),ROOT/'scripts/audit_native_current_state_dispatch_v1.py',"
        "ROOT/'scripts/native_current_state_recovery_v1.py',ROOT/'scripts/current_state_recovery_service_v1.py',"
        "ROOT/'scripts/current_state_recovery_planner_v1.py',ROOT/'scripts/check_early_support_dispatch_v1.py',")
    ast.parse(text);audit.write_text(text)
    result.update(entry_sha256=sha(entry),audit_sha256=sha(audit),
        bundle=str(a.bundle_output.resolve()),bundle_manifest_sha256=sha(a.bundle_output/'manifest.json'),
        source_sha256={**result['source_sha256'],**m['source_sha256']},
        scope=m['scope'],native_execution=False,teacher_acceptance=False,
        planning_schedule=m['current_state_planning'],interface_check=str(a.interface_check.resolve()))
    (a.output/'preparation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(entry=result['entry'],audit=result['audit'],bundle=result['bundle'])))


if __name__=='__main__':main()
