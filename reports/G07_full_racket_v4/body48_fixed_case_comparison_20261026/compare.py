"""Compare complete real fixed TRAIN128 policies without opening formal data."""
import argparse
from collections import Counter
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases,summarize_training
from quadsmash.evaluation_v4 import independent_episode,FAULTS


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def load(run,policy,protocol):
    job,inputs,metrics=[read(run/name) for name in ['job.json','INPUTS.json','metrics.json']]
    assert job['status']=='SUCCEEDED' and job['exit_code']==0 and job['receipt_present']
    steps=read(run/'completed.json')['env_steps']
    assert 0<steps<=102400 and job['charged_env_steps']==102400
    assert metrics['actual_control_env_steps']==steps
    assert inputs['policy_sha256']==metrics['policy_sha256']==sha(policy)
    assert inputs['protocol_sha256']==metrics['protocol_sha256']==sha(ROOT/'configs/g07_full_racket_eval_v4_r3.json')
    assert inputs['training_indices']==list(range(128)) and metrics['noise_level']=='none'
    assert inputs['reset_noise_seed']==metrics['noise_seed']==724201
    assert all(value==0 for value in inputs['reset_noise_amplitudes'].values())
    for name in ['quadsmash/evaluation_v4.py','quadsmash/control/full_racket_train_evaluation_v4.py',
                 'quadsmash/physics/rally_rules_v4.py']:
        assert sha(ROOT/name)==read(run/'source_hashes.json')[name]
    cases,indices=training_cases(ROOT,protocol)
    rows=read(run/'episodes.json')
    summary=summarize_training(rows,cases,indices,protocol)
    assert all(metrics[key]==value for key,value in summary.items())
    assert summary['numerical_failures']==0
    return rows,summary


def detail(row,protocol):
    outcome=independent_episode(row,protocol)
    velocity=np.asarray(row['outgoing_velocity_mps'],dtype=float)
    attack=bool(outcome['airborne_contact'] and row['released'] and np.isfinite(velocity).all()
        and np.linalg.norm(velocity)>=8. and velocity[0]>0 and velocity[2]<=-1.)
    valid_cross=bool(attack and row['crossed_net'] and not outcome['fault'])
    target=bool(valid_cross and row['landed_target'])
    times=[row[key] for key in ['contact_time_s','release_time_s','crossing_time_s','first_landing_time_s']]
    ordered=bool(np.isfinite(times).all() and times[0]>=protocol['preparation_seconds']-protocol['physics_dt_s']
        and times[1]>=times[0] and times[2]+protocol['physics_dt_s']>=times[0]
        and times[3]+protocol['physics_dt_s']>=max(times[1],times[2]))
    if outcome['complete']:
        reason='complete'
    elif not outcome['airborne_contact']:
        reason=('no_racket_contact' if not row['had_contact'] else
                'frame_only' if row['frame_hit'] and not row['string_hit'] else 'other_no_legal_airborne_strings')
    elif not attack:
        reason='attack_requirements_missed'
    elif outcome['fault']:
        reason='rally_fault'
    elif not row['crossed_net']:
        reason='no_crossing'
    elif not ordered:
        reason='event_order'
    elif not row['landed_target']:
        reason='outside_original_target'
    elif not outcome['conditional_recovery']:
        reason='recovery'
    else:
        reason='other_incomplete'
    return dict(air=int(outcome['airborne_contact']),recovery=int(outcome['conditional_recovery']),
        complete=int(outcome['complete']),attack=int(attack),valid_attacking_cross=int(valid_cross),
        valid_attack_cross_target=int(target),reason=reason,
        faults=';'.join(name for name in FAULTS if row[name]),racket_contact=int(row['had_contact']),
        string_hit=int(row['string_hit']),frame_hit=int(row['frame_hit']),
        contact_x=row['contact_point_m'][0],contact_y=row['contact_point_m'][1],contact_z=row['contact_point_m'][2],
        outgoing_speed=float(np.linalg.norm(velocity)) if np.isfinite(velocity).all() else None,
        outgoing_vx=row['outgoing_velocity_mps'][0],outgoing_vz=row['outgoing_velocity_mps'][2],
        first_land_x=row['first_landing_m'][0],first_land_y=row['first_landing_m'][1],
        stable_recovery_s=row['jump']['recovery_s'],body_landing=int(row['jump']['body_landing']))


def plot(rows,details,court,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    figure,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for column,(label,raw,checked) in enumerate(zip(['Source teacher','Frozen candidate'],rows,details)):
        air=[i for i,item in enumerate(checked) if item['air']]
        ax=axes[0,column]
        for complete,color in [(0,'#d9852f'),(1,'#23865d')]:
            selected=[i for i in air if checked[i]['complete']==complete]
            ax.scatter([raw[i]['contact_point_m'][0] for i in selected],
                [raw[i]['contact_point_m'][2] for i in selected],c=color,
                marker='*' if complete else 'o',s=110 if complete else 24,
                label='Complete' if complete else 'Incomplete')
        ax.axvline(court['net_plane_x_m'],color='#555555',linestyle='--',label='Net plane')
        ax.axhline(court['net_center_top_m'],color='#888888',linestyle=':')
        ax.set(title=f'{label}: legal airborne string contacts ({len(air)})',xlabel='Actual contact x (m)',ylabel='Actual contact z (m)')
        ax.legend(fontsize=8)
        ax=axes[1,column]
        selected=[i for i,item in enumerate(checked) if item['valid_attacking_cross']]
        for complete,color in [(0,'#d9852f'),(1,'#23865d')]:
            ids=[i for i in selected if checked[i]['complete']==complete]
            ax.scatter([raw[i]['first_landing_m'][0] for i in ids],[raw[i]['first_landing_m'][1] for i in ids],
                c=color,marker='*' if complete else 'o',s=110 if complete else 24)
        low,high=court['opponent_x_limits_m']
        ax.add_patch(Rectangle((low,-court['singles_width_m']/2),high-low,court['singles_width_m'],
            fill=False,color='#666666',label='Legal opponent singles half'))
        low,high=court['target_x_limits_m']
        ax.add_patch(Rectangle((low,-court['target_y_half_width_m']),high-low,2*court['target_y_half_width_m'],
            fill=False,color='#235c9e',linewidth=2,label='Original target'))
        ax.set(title=f'Valid attacking crossings: actual first landings ({len(selected)})',xlabel='First landing x (m)',ylabel='First landing y (m)')
        ax.autoscale_view()
        ax.legend(fontsize=8)
    figure.suptitle('Actual fixed TRAIN128 observations; one native run per policy, original scores retained')
    figure.savefig(output,dpi=150)
    plt.close(figure)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run',type=Path,required=True)
    parser.add_argument('--reference-policy',type=Path,required=True)
    parser.add_argument('--candidate-run',type=Path,required=True)
    parser.add_argument('--candidate-policy',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists()
    protocol_path=ROOT/'configs/g07_full_racket_eval_v4_r3.json'
    protocol=read(protocol_path)
    protocol['_root']=str(ROOT)
    for name,digest in protocol['physical_configs'].items():
        assert sha(ROOT/name)==digest
    data=[load(run,policy,protocol) for run,policy in
        [(args.reference_run,args.reference_policy),(args.candidate_run,args.candidate_policy)]]
    rows=[item[0] for item in data]
    details=[[detail(row,protocol) for row in group] for group in rows]
    assert all(a['case_id']==b['case_id'] for a,b in zip(*rows))
    table=[dict(case_id=a['case_id'],training_index=a['source_training_index'],
        **{'source_'+key:value for key,value in d.items()},**{'candidate_'+key:value for key,value in e.items()})
        for a,b,d,e in zip(*rows,*details)]
    for row in table:
        for key,value in row.items():
            if isinstance(value,(float,np.floating)) and not np.isfinite(value):
                row[key]=None
    sources=[Path(__file__),protocol_path,args.reference_policy,args.candidate_policy]
    for run in [args.reference_run,args.candidate_run]:
        sources.extend(run/name for name in ['INPUTS.json','metrics.json','episodes.json','job.json','completed.json','source_hashes.json'])
    args.output.mkdir(parents=True)
    (args.output/'PLAN.json').write_text(json.dumps(dict(created_at=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(path.resolve()):sha(path) for path in sources},scope='Actual original fixed TRAIN128 paired records only.',
        formal_cases_read=False,new_native_execution=False),indent=2)+'\n')
    with (args.output/'PAIRED_CASES.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(table)
    transitions={}
    for key in ['air','recovery','attack','valid_attacking_cross','valid_attack_cross_target','complete']:
        transitions[key]=dict(gained=[i for i,(a,b) in enumerate(zip(*details)) if b[key] and not a[key]],
            lost=[i for i,(a,b) in enumerate(zip(*details)) if a[key] and not b[key]])
    counts=[{key:sum(row[key] for row in group) for key in
        ['air','attack','valid_attacking_cross','valid_attack_cross_target','recovery','complete']} for group in details]
    assert all(count['complete']==item[1]['complete'] for count,item in zip(counts,data))
    result=dict(checked_at=datetime.now(timezone.utc).isoformat(),reference_summary=data[0][1],candidate_summary=data[1][1],
        funnels=counts,exclusive_final_failure_reasons=[dict(Counter(row['reason'] for row in group)) for group in details],
        paired_transitions=transitions,original_target_unchanged=True,
        caution='One fixed native run per policy. Paired case identities are exact; differences do not prove cross-run statistical significance or native causality.',
        formal_cases_read=False,new_native_execution=False,formal_acceptance=False,goal_complete=False)
    (args.output/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    plot(rows,details,read(ROOT/'configs/court_net_v4.json'),args.output/'CONTACTS_AND_FIRST_LANDINGS.png')
    print(json.dumps(dict(funnels=counts,complete_changes=transitions['complete'])),flush=True)


if __name__=='__main__':
    main()
