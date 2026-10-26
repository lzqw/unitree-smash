"""Export actual candidate search episodes; preserve native generation/episode IDs."""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import numpy as np

MAIN=Path(__file__).resolve().parents[1].parent/'unitree_smash'
sys.path.insert(0,str(MAIN))
from quadsmash.evaluation_v4 import independent_episode,FAULTS
from quadsmash.control.full_racket_body_search_v4 import summarize_candidate


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output-prefix',type=Path,required=True);args=p.parse_args()
    output=args.output_prefix.with_suffix('.csv');manifest=args.output_prefix.with_suffix('.json')
    assert not output.exists() and not manifest.exists()
    run=args.run.resolve();metrics=read(run/'metrics.json');assert read(run/'job.json')['status']=='SUCCEEDED'
    assert read(run/'completed.json')['env_steps']==metrics['actual_control_env_steps']==2048000
    protocol_path=MAIN/'configs/g07_full_racket_eval_v4_r3.json';protocol=read(protocol_path);protocol['_root']=str(MAIN)
    assert read(run/'INPUTS.json')['protocol_sha256']==sha(protocol_path)
    sources=[Path(__file__),protocol_path,run/'job.json',run/'completed.json',run/'metrics.json',run/'INPUTS.json']
    exported=[];counts=dict(episodes=0,airborne_contacts=0,conditional_recovery_count=0,complete=0,numerical_failures=0)
    for entry in metrics['history']:
        generation=entry['generation'];rows_path=run/f'episodes_{generation:03d}.json';rows=read(rows_path)
        ground_path=run/f'initial_ground_states_{generation:03d}.json';ground=read(ground_path);sources.extend([rows_path,ground_path])
        assert len(rows)==1280
        actual=[summarize_candidate(rows[i*128:(i+1)*128],protocol) for i in range(10)]
        assert actual==entry['summaries']
        for i,row in enumerate(rows):
            assert row['env_id']==i and row['candidate_id']==i//128 and row['source_training_index']==i%128
            assert row['episode_index']==ground['episode_ids'][i] and row['ground_init']
            checked=independent_episode(row,protocol)
            flat=dict(record_scope='fixed_candidate_native_search_episode',generation=generation)
            for name in ['candidate_id','case_id','source_training_index','env_id','episode_index','ground_init',
                'had_contact','legal_initial_contact','string_hit','frame_hit','airborne_string_hit','released',
                'crossed_net','landed_legal_court','landed_target','contact_time_s','release_time_s',
                'crossing_time_s','first_landing_time_s','net_margin_m','numerical_failures',*FAULTS]:flat[name]=row[name]
            flat.update({f'jump_{name}':row['jump'][name] for name in ['airborne_s','recovery_s',
                'takeoff_time_s','landing_time_s','qualified_jump_landed','stable_recovery','body_landing',
                'terminated','maximum_body_ground_force_n','max_leg_torque_nm','max_arm_torque_nm','max_joint_speed_rad_s']})
            flat.update({f'canonical_{name}':value for name,value in checked.items()})
            for name in ['contact_point_m','outgoing_velocity_mps','crossing_point_m','first_landing_m']:
                flat.update({f'{name}_{axis}':value for axis,value in zip('xyz',row[name])})
            velocity=np.asarray(row['outgoing_velocity_mps']);flat['outgoing_ball_speed_mps']=float(np.linalg.norm(velocity))
            for name,value in flat.items():
                if isinstance(value,(bool,np.bool_)):flat[name]=int(value)
                elif value is None or isinstance(value,(float,np.floating)) and not np.isfinite(value):flat[name]=''
            exported.append(flat);counts['episodes']+=1;counts['airborne_contacts']+=checked['airborne_contact']
            counts['conditional_recovery_count']+=checked['conditional_recovery'];counts['complete']+=checked['complete']
            counts['numerical_failures']+=row['numerical_failures']
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(exported[0]),lineterminator='\n')
    writer.writeheader();writer.writerows(exported);table=list(csv.DictReader(io.StringIO(stream.getvalue())))
    assert len(table)==counts['episodes']==2560 and sum(int(r['canonical_complete']) for r in table)==counts['complete']==1
    output.parent.mkdir(parents=True,exist_ok=True);output.with_suffix('.tmp').write_text(stream.getvalue());output.with_suffix('.tmp').replace(output)
    data=dict(checked_at=datetime.now(timezone.utc).isoformat(),source_sha256={str(path):sha(path) for path in sources},
        csv=str(output),csv_sha256=sha(output),counts=counts,all_native_episode_ids_preserved=True,
        all20_candidate_summaries_match_native=True,missing_events='Empty cells; never replaced by zero.',
        scope='Two generations,10 fixed candidates each,128 cases each. Aggregate counts are not one fixed-policy score.',
        new_native_execution=False,formal_cases_read=False,formal_acceptance=False)
    manifest.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(dict(csv=str(output),counts=counts)),flush=True)


if __name__=='__main__':main()
