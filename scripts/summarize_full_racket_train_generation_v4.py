#!/usr/bin/env python3
"""Recheck a completed native TRAIN generation using frozen episode criteria."""
import argparse,csv,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from quadsmash.evaluation_v4 import independent_episode
from quadsmash.control.full_racket_contact_curriculum_v4 import usable_air_contact


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--generation',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Use a fresh report path')
    suffix=f'{a.generation:03d}';names=[f'episodes_{suffix}.json',f'population_{suffix}.pt',
        f'public_and_poses_{suffix}.npz',f'all_first_episodes_{suffix}.npz','source_hashes.json']
    rows=json.loads((a.run/names[0]).read_text());state=torch.load(a.run/names[1],map_location='cpu',weights_only=False)
    cases=state['training_case_indices'];count=state['candidate_count']*len(cases)
    assert len(rows)==count and [r['env_id'] for r in rows]==list(range(count))
    assert all(r['episode_index']==0 and r['source_training_index']==cases[r['env_id']%len(cases)] for r in rows)
    assert json.dumps(rows,sort_keys=True)==json.dumps(state['rows'],sort_keys=True)
    protocol_path=ROOT/'configs/g07_full_racket_eval_v4_r3.json';protocol=json.loads(protocol_path.read_text())
    assert hashlib.sha256(protocol_path.read_bytes()).hexdigest()==state['protocol_sha256']
    assert state['asset_sha256']==protocol['asset']['sha256']
    protocol['_root']=str(ROOT)
    with np.load(a.run/f'all_first_episodes_{suffix}.npz') as native:
        initial=native['link_poses'][0,:,0,:3]
        assert np.max(abs(initial[:,2]-protocol['initialization']['root_position_m'][2]))<1e-6
        assert np.max(abs(initial[:,:2]))<1e-6
    with np.load(a.run/f'public_and_poses_{suffix}.npz') as public:
        ticks=len(public['actions']);assert np.isfinite(public['actions']).all() and (abs(public['actions'])<=1).all()
    checked=[]
    for row in rows:
        value=independent_episode(dict(row,case_id=f'TRAIN{row["source_training_index"]}_G{a.generation}_ENV{row["env_id"]}',ground_init=True),protocol)
        velocity=np.array(row['outgoing_velocity_mps'])
        attack=bool(value['airborne_contact'] and row['released'] and np.isfinite(velocity).all()
                    and np.linalg.norm(velocity)>=8 and velocity[0]>0 and velocity[2]<=-1)
        assert row['attack_outgoing']==attack
        checked.append(value)
    keys=['had_contact','frame_hit','airborne_string_hit','attack_outgoing','recovery_success',
          'crossed_net','landed_legal_court','landed_target','complete_smash','robot_net_fault','racket_invasion_fault']
    counts={k:sum(bool(r[k]) for r in rows) for k in keys}
    counts.update(executions=count,usable_air_strings=sum(usable_air_contact(r) for r in rows),
        air_strings_and_recovery=sum(r['airborne_string_hit'] and r['recovery_success'] for r in rows),
        attack_outgoing_and_recovery=sum(r['attack_outgoing'] and r['recovery_success'] for r in rows),
        canonical_air_contacts=sum(v['airborne_contact'] for v in checked),
        canonical_conditional_recovery_count=sum(v['conditional_recovery'] for v in checked),
        canonical_complete=sum(v['complete'] for v in checked),
        robot_rule_terminal=sum(r['training_terminal_reasons']['robot_rule_fault'] for r in rows),
        body_ground_terminal=sum(r['training_terminal_reasons']['body_ground'] for r in rows),
        numerical_failures=sum(r['numerical_failures'] for r in rows))
    per_candidate=[]
    for candidate in range(state['candidate_count']):
        indices=[i for i,r in enumerate(rows) if r['candidate_id']==candidate]
        assert len(indices)==len(cases)
        group=[rows[i] for i in indices];canonical=[checked[i] for i in indices]
        values={key:sum(bool(r[key]) for r in group) for key in keys}
        values.update(canonical_air_contacts=sum(v['airborne_contact'] for v in canonical),
            canonical_conditional_recovery_count=sum(v['conditional_recovery'] for v in canonical),
            canonical_complete=sum(v['complete'] for v in canonical),
            canonical_attack_and_recovery=sum(r['attack_outgoing'] and v['conditional_recovery'] for r,v in zip(group,canonical)),
            canonical_target_and_recovery=sum(r['landed_target'] and v['conditional_recovery'] for r,v in zip(group,canonical)),
            body_ground_terminal=sum(r['training_terminal_reasons']['body_ground'] for r in group),
            robot_rule_terminal=sum(r['training_terminal_reasons']['robot_rule_fault'] for r in group),
            numerical_failures=sum(r['numerical_failures'] for r in group))
        air=values['canonical_air_contacts'];recovered=values['canonical_conditional_recovery_count']
        per_candidate.append(dict(candidate_id=candidate,executions=len(group),counts=values,
            airborne_contact_rate=air/len(group),conditional_recovery_rate=recovered/air if air else 0.,
            complete_success_rate=values['canonical_complete']/len(group)))
    for key in ['canonical_air_contacts','canonical_conditional_recovery_count','canonical_complete',
                'had_contact','attack_outgoing','landed_target','numerical_failures']:
        assert sum(v['counts'][key] for v in per_candidate)==counts[key]
    report=dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='Completed native TRAIN generation only. Canonical per-episode criteria recomputed on TRAINrows without reading formal panel cases; not a formal panel result or independent success rate.',
        run=a.run.name,generation=a.generation,candidates=state['candidate_count'],distinct_TRAIN_indices=cases,
        distinct_cases=len(cases),control_ticks=ticks,actual_control_env_steps=ticks*count,counts=counts,
        per_candidate=per_candidate,
        best_candidate_indices=[r['candidate_id'] for r in state['best_result']],
        all_native_and_independent_complete_flags_agree=True,all_attack_gates_recomputed_from_actual_outgoing=True,
        all_first_ground_initializations_verified=True,whole_job_complete=(a.run/'completed.json').exists(),
        native_execution=True,formal_acceptance=False,
        source_sha256={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in
                       [a.run/n for n in names]+[protocol_path,ROOT/'quadsmash/evaluation_v4.py',Path(__file__)]})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    columns=['env_id','candidate_id','source_training_index','airborne_string_hit','attack_outgoing',
             'recovery_success','landed_legal_court','landed_target','complete_smash','robot_net_fault',
             'recovery_s','qualified_jump_landed','maximum_body_ground_force_n','first_landing_x_m','first_landing_y_m']
    with a.output.with_suffix('.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,lineterminator='\n');writer.writeheader()
        for row in rows:
            values={key:row[key] for key in columns[:10]}
            values.update({key:row['jump'][key] for key in columns[10:13]})
            values.update(first_landing_x_m=row['first_landing_m'][0],first_landing_y_m=row['first_landing_m'][1]);writer.writerow(values)
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))


if __name__=='__main__':main()
