"""Admit archived initial references after current-state V15 model feedback and preservation.

Input/wiring evidence only. Native success is not inferred from this check.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['parent-plan','entry','configuration','coverage','preservation','feedback','end-check','output']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();read=lambda f:json.loads(Path(f).read_text());sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    if a.output.exists():raise ValueError('Fresh admission receipt required')
    parent=read(a.parent_plan);cmd=parent['command']
    old_entry=next(Path(x) for x in cmd if x.endswith('/evaluate_pure_control.py'))
    old_cfg=read(cmd[cmd.index('--pure-control-configuration')+1]);cfg=read(a.configuration)
    if sha(a.entry)!=sha(old_entry):raise ValueError('Native entry differs from verified V15')
    text=a.entry.read_text();ast.parse(text)
    call='action=native_replay.action(tick,obs,action,finished)'
    if text.count(call)!=1 or 'native_pure_control_stance_foot_reference_v15' not in text:raise ValueError('Wrong final owner')
    block=text[text.index(call):text.index('observation, _, done, _ = wrapped.step(action)')]
    if 'action = ' in block or 'action=' in block.split('\n',1)[1]:raise ValueError('Second action writer')
    for key in set(old_cfg)|set(cfg):
        if key not in ['method','model_reference_library'] and cfg.get(key)!=old_cfg.get(key):
            raise ValueError('Changed execution setting: '+key)
    library=Path(cfg['model_reference_library']);bank=read(library);base=read(old_cfg['model_reference_library'])
    if bank['references'][:len(base['references'])]!=base['references']:
        raise ValueError('Old reference bank changed')
    new=bank['references'][len(base['references']):]
    if not new or bank['native_success'] or bank['valid_expert_labels']:raise ValueError('Model-only additions required')
    coverage=read(a.coverage);preservation=read(a.preservation);inputs=read(a.feedback/'inputs.json')
    feedback=read(a.feedback/'result.json');end=read(a.end_check)
    if (feedback['state']!='COMPLETED_ARCHIVED_REFERENCE_RECHECK' or inputs['physical_substeps']!=160
            or inputs['controller']!='V15' or inputs['scoring_substeps']!=16):
        raise ValueError('Completed fine-model feedback required')
    if not coverage['passed'] or coverage['native_execution'] or coverage['runtime_gate_uses_case_ids']:
        raise ValueError('Causal coverage evidence required')
    if not preservation['passed'] or preservation['native_execution'] or preservation['native_success'] or preservation['valid_expert_labels']:
        raise ValueError('Complete model-only preservation required')
    pins={}
    if end['acceptance_changed'] or end['native_success'] or end['valid_expert_labels']:
        raise ValueError('Invalid end-observation contract')
    for data in [parent,bank,coverage,preservation,inputs,end]:
        for f,h in data['source_sha256'].items():
            if sha(f)!=h or (f in pins and pins[f]!=h):raise ValueError('Changed/conflicting source '+f)
            pins[f]=h
    for data in [coverage,preservation]:
        if data['source_sha256'].get(str(library.resolve()))!=sha(library):raise ValueError('Checked another expanded bank')
    baseline_paths=[Path(f) for f in preservation['source_sha256'] if Path(f).name=='best_verified_v15.json']
    if len(baseline_paths)!=1:raise ValueError('Missing native baseline identity')
    protected=set(read(baseline_paths[0])['clean_stable_cases'])
    if {r['case'] for r in preservation['checks']}!=protected or not protected:
        raise ValueError('Missing protected success')
    for row in preservation['checks']:
        if row['last_tick']!=799 or row['commands_checked']!=800-row['first_tick'] or row['maximum_request_error_rad']>2e-6:
            raise ValueError('Incomplete unchanged native command history')
    matches={r['source_key']:r for r in coverage['new_matches']}
    if set(matches)!={r['source_key'] for r in new}:raise ValueError('Unverified new match')
    checks=[]
    for row in new:
        match=matches[row['source_key']];key=match['feedback_state_key']
        if match['first_command_error_rad']>2e-6:raise ValueError('Different entry command')
        found=[r for r in feedback['results'] if r['state_key']==key]
        final=[r for r in end['results'] if r['state_key']==key]
        if len(found)!=1 or len(final)!=1:raise ValueError('Missing complete result')
        result=found[0];last=final[0]
        if result['failure'] is not None or result['steps']!=800-match['actual_tick']:
            raise ValueError('Incomplete model feedback')
        if not last['complete_model_feedback'] or not last['model_terminal_window_passed'] or last['end_time_s']!=8. or last['end_observation_stable_s']<.8:
            raise ValueError('Incomplete final stability observation')
        checks.append(dict(match=match,end_observation=last))
    files=[Path(__file__),a.parent_plan,a.entry,a.configuration,a.coverage,a.preservation,
        a.feedback/'inputs.json',a.feedback/'result.json',a.end_check,library]
    pins.update({str(f.resolve()):sha(f) for f in files})
    receipt=dict(passed=True,new_reference_checks=checks,preserved_commands=preservation['checks'],
        source_sha256=pins,entry_sha256=sha(a.entry),native_execution=False,
        native_success=False,valid_expert_labels=False,physical_interface_unchanged=True,
        single_final_writer=True,model_pass_is_not_native_acceptance=True)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(passed=True,new_references=len(checks),protected_cases=len(protected),native_execution=False)))


if __name__=='__main__':main()
