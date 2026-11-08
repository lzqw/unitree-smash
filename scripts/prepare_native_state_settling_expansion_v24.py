"""Prepare/verify a V23 native test with one additional feedback-checked reference."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.retry_budgeted_settling_feedback_v1 import sha,write


def validate(plan_path, output):
    read=lambda f:json.loads(Path(f).read_text())
    if output.exists():raise ValueError('Fresh wiring receipt required')
    plan=read(plan_path);parent=read(plan['parent_plan'])
    for path,digest in plan['source_sha256'].items():
        if sha(path)!=digest:raise ValueError('Changed prepared input '+path)
    old_command=parent['command'];command=plan['command']
    old_entry=next(Path(x) for x in old_command if x.endswith('/evaluate_pure_control.py'))
    entry=next(Path(x) for x in command if x.endswith('/evaluate_pure_control.py'))
    if sha(entry)!=sha(old_entry):raise ValueError('Changed V23 native entry')
    text=entry.read_text();call='action=native_replay.action(tick,obs,action,finished)'
    if text.count(call)!=1 or 'native_pure_control_state_settling_v23' not in text:
        raise ValueError('Wrong final controller owner')
    old_cfg=read(old_command[old_command.index('--pure-control-configuration')+1])
    cfg=read(command[command.index('--pure-control-configuration')+1])
    if cfg['controller']!='pure_control_state_settling_v23':raise ValueError('Reference requires V23')
    for key in set(old_cfg)|set(cfg):
        if key not in ['method','model_reference_library'] and cfg.get(key)!=old_cfg.get(key):
            raise ValueError('Changed physical/control setting '+key)
    if command[command.index('--max-seconds')+1]!='7200' or command[command.index('--env-steps-cap')+1]!='102400':
        raise ValueError('Changed declared native bounds')
    bank=read(cfg['model_reference_library']);old_bank=read(old_cfg['model_reference_library'])
    if bank['references'][:-1]!=old_bank['references'] or bank['requires_controller']!='PureControlStateSettlingV23':
        raise ValueError('Changed original bank or controller dependency')
    receipt=read(plan['bank_receipt']);added=bank['references'][-1]
    if not receipt['passed'] or receipt['native_execution'] or receipt['native_success'] or receipt['valid_expert_labels']:
        raise ValueError('Invalid preparation evidence')
    if receipt['causal_entries_checked']!=110 or len(receipt['new_matches'])!=1:
        raise ValueError('Missing causal coverage audit')
    feedback=read(added['source_feedback']);initial=feedback['initial_actual_state']
    if (feedback['mode']!='state_settling' or feedback['candidate_scoring_substeps']!=16 or feedback['physical_substeps']!=160
            or feedback['failure'] is not None or not feedback['full_model_path']
            or feedback['steps']!=800-initial['tick'] or feedback['final_model_stable_s']<.8
            or feedback['native_success'] or feedback['valid_expert_labels']):
        raise ValueError('Incomplete actual-controller model feedback')
    for data in [bank,receipt]:
        for path,digest in data['source_sha256'].items():
            if plan['source_sha256'].get(path)!=digest or sha(path)!=digest:
                raise ValueError('Unpinned bank or coverage evidence '+path)
    result=dict(passed=True,controller='PureControlStateSettlingV23',reference_count=len(bank['references']),
        new_matches=receipt['new_matches'],reference_open_loop_complete=added['reference_open_loop_complete'],
        full_closed_loop_model_path=True,model_stable_s=feedback['final_model_stable_s'],
        hardware_and_prefix_unchanged=True,native_execution=False,native_success=False,valid_expert_labels=False,
        old_native_success_retention_unproven=True,source_sha256=plan['source_sha256'])
    write(output,result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify-plan',type=Path);p.add_argument('--receipt',type=Path)
    for name in ['parent-plan','bank-preparation','output','native-output']:
        p.add_argument('--'+name,type=Path)
    a=p.parse_args()
    if a.verify_plan:
        if not a.receipt:raise ValueError('Receipt output required')
        print(json.dumps({k:v for k,v in validate(a.verify_plan,a.receipt).items() if k!='source_sha256'}));return
    if any(x is None for x in [a.parent_plan,a.bank_preparation,a.output,a.native_output]):
        raise ValueError('Complete preparation arguments required')
    if a.output.exists() or a.native_output.exists():raise ValueError('Fresh preparation required')
    read=lambda f:json.loads(Path(f).read_text());plan=read(a.parent_plan)
    for path,digest in plan['source_sha256'].items():
        if sha(path)!=digest:raise ValueError('Changed parent input '+path)
    bank_receipt=a.bank_preparation/'receipt.json';receipt=read(bank_receipt)
    for path,digest in receipt['source_sha256'].items():
        if sha(path)!=digest or (path in plan['source_sha256'] and plan['source_sha256'][path]!=digest):
            raise ValueError('Changed/conflicting expansion input '+path)
    command=plan['command'];old=next(Path(x) for x in command if x.endswith('/evaluate_pure_control.py'))
    cfg=read(command[command.index('--pure-control-configuration')+1])
    cfg.update(model_reference_library=str((a.bank_preparation/'reference_library.json').resolve()),
        method='V23 state-driven support with one additional partial reference backed by complete model feedback')
    a.output.mkdir(parents=True);entry=a.output/'evaluate_pure_control.py';entry.write_bytes(old.read_bytes())
    config=a.output/'controller.json';write(config,cfg)
    command=[str(entry.resolve()) if x==str(old) else x for x in command]
    for i,x in enumerate(command[:-1]):
        if x=='--output':command[i+1]=str(a.native_output.resolve())
        elif x=='--pure-control-configuration':command[i+1]=str(config.resolve())
    pins=plan['source_sha256'];pins.update(receipt['source_sha256'])
    pins.update({str(f.resolve()):sha(f) for f in [Path(__file__),a.parent_plan,bank_receipt,entry,config]})
    target=a.output/'plan.json'
    plan.update(command=command,output=str(a.native_output.resolve()),parent_plan=str(a.parent_plan.resolve()),
        bank_receipt=str(bank_receipt.resolve()),source_sha256=pins,entry_sha256=sha(entry),
        scope=__doc__,declared_max_native_seconds=7200,model_accepted=False,training=False,valid_expert_labels=False,
        old_success_retention_unproven=True,
        check_command=['bash','scripts/python.sh',str(Path(__file__).resolve()),'--verify-plan',str(target.resolve()),
            '--receipt',str((a.output/'entry_check.json').resolve())],
        post_command=['bash','scripts/python.sh','scripts/audit_native_pure_control_v2.py',
            '--run',str(a.native_output.resolve()),'--output',str((a.output/'native_audit.json').resolve())])
    write(target,plan);prepared=a.output/'prepared_entry_check.json';validate(target,prepared)
    plan['source_sha256'][str(prepared.resolve())]=sha(prepared);write(target,plan)
    print(json.dumps(dict(plan=str(target),native_execution=False,pinned_files=len(plan['source_sha256']))))


if __name__=='__main__':main()
