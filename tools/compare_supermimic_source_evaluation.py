"""Compare one completed source-policy evaluation with preserved TRAIN128.

Reads only files. Never launches a simulator, modifies scores, fills missing
contact evidence with zero, or promotes a partial run to a completed run.
"""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(run, baseline):
    run, baseline = Path(run), Path(baseline)
    completed = read(run/'completed.json')
    if completed['control_ticks'] != 800 or completed['episodes'] != 128:
        raise ValueError('Require a complete800-tick128-case evaluation')
    current = read(run/'metrics.json')
    old = read(baseline/'metrics.json')
    inputs = read(baseline/'INPUTS.json')
    if current['evaluated_checkpoint_sha256'] != inputs['policy_sha256']:
        raise ValueError('This zero-update comparison requires the same source checkpoint')
    if current['actor_updated']:
        raise ValueError('Not a zero-update source evaluation')
    if current['num_envs'] != 128 or current['control_ticks'] != 800:
        raise ValueError('Metrics do not describe the completed panel')
    rows = read(run/'episodes_summary.json')
    old_rows = old['complete5_case_outcomes']
    if len(rows) != 128 or len(old_rows) != 128:
        raise ValueError('Require128 explicit case outcomes from each version')
    if [r['source_case_index'] for r in rows] != list(range(128)):
        raise ValueError('Current case order changed')
    if any(r['env_id'] != r['source_case_index'] or r['episode_index'] != 0 for r in rows):
        raise ValueError('Current lane mapping or first-episode association changed')
    if sorted(r['training_index'] for r in old_rows) != list(range(128)):
        raise ValueError('Historical case mapping incomplete')
    names = {'air': 'qualified_airborne_string_contact',
             'ball5_except_recovery': 'ball_side_complete_except_recovery',
             'recovery': 'recovery_success', 'complete5': 'complete5'}
    a = {k:{r['training_index'] for r in old_rows if r[k]} for k in names}
    b = {k:{r['source_case_index'] for r in rows if r[field]} for k,field in names.items()}
    for label, sets in [('historical',a),('current',b)]:
        if not sets['complete5'] <= (sets['air'] & sets['recovery'] & sets['ball5_except_recovery']):
            raise ValueError(f'{label} complete5 contradicts its component outcomes')
    # Conditional recovery uses fixed historical qualified-contact cases.
    original_recovery = a['recovery'] & a['air']
    current_recovery = b['recovery'] & a['air']
    for field,key in [('qualified_airborne_string_contact','air'),
                      ('ball_side_complete_except_recovery','ball5_except_recovery'),
                      ('complete5','complete5')]:
        if current[field] != len(b[key]):
            raise ValueError(f'Metrics/rows conflict: {field}')
    if old['airborne_contacts'] != len(a['air']) or old['complete5'] != len(a['complete5']):
        raise ValueError('Historical metrics/rows conflict')
    mapping = read(run/'command_consistency.json')
    command_ok = (mapping['checked_native_ticks']==800
                  and mapping['tolerance_rad']==2e-6
                  and mapping['max_final_target_error_rad']<=2e-6
                  and all(n>0 for n in mapping['phase_samples'].values()))
    same_sets = all(a[k]==b[k] for k in names)
    exit_code = int((run.parent/'exit_code.txt').read_text().strip())
    changes = {k:dict(baseline=sorted(a[k]),current=sorted(b[k]),
                     retained=sorted(a[k]&b[k]),lost=sorted(a[k]-b[k]),added=sorted(b[k]-a[k])) for k in names}
    recovery = dict(denominator=len(a['air']),baseline=len(original_recovery),current=len(current_recovery),
                    retained=sorted(original_recovery & current_recovery),
                    lost=sorted(original_recovery-current_recovery),added=sorted(current_recovery-original_recovery))
    assert recovery['current']==recovery['baseline']+len(recovery['added'])-len(recovery['lost'])
    clean = {}
    for key in ('head_contact','independent_nonfoot_contact'):
        applicable=[r for r in rows if r['source_case_index'] in a['air'] and r['recovery_success']]
        clean[key] = (None if any(r.get(key) is None for r in applicable)
                      else sum(not r[key] for r in applicable))
    return dict(scope='TRAIN128 case outcomes and same-live-input command comparison; not historical trajectory identity, generalization or A/B acceptance',
        run=str(run), baseline=str(baseline), source_sha256=current['evaluated_checkpoint_sha256'],
        files={str(p):digest(p) for p in [run/'metrics.json',run/'episodes_summary.json',
               run/'completed.json',baseline/'metrics.json',baseline/'INPUTS.json']},
        exit_code=exit_code, command_mapping_passed=command_ok,
        matched_outcome_sets=same_sets, changes=changes, fixed_contact_recovery=recovery,
        current_contact_conditional_recovery=dict(count=current['conditional_recovery_count'],
            denominator=current['conditional_recovery_denominator']),
        no_contact_recovery=clean, numerical_audit_count=current['numerical_failures'],
        task_contract=current['task_contract'],
        recorded_outcomes_and_commands_match=(same_sets and command_ok and exit_code==0 and current['numerical_failures']==0))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=compare(args.run,args.baseline)
    # Refuse to replace historical results.
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({k:result[k] for k in ('exit_code','command_mapping_passed',
        'matched_outcome_sets','fixed_contact_recovery','no_contact_recovery',
        'numerical_audit_count','recorded_outcomes_and_commands_match')},indent=2))


if __name__=='__main__':
    main()
