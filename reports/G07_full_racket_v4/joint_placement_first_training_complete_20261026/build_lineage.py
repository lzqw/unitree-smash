"""Preserve exact episode timing and changing-actor lineage of joint TRAIN hits."""
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
RUN=ROOT/'reports/G07_full_racket_v4/residual_ppo_joint_placement_actor32_n2048_resume32_01'
INITIAL_ITERATION=8
sys.path.insert(0,str(ROOT))
from quadsmash.evaluation_v4 import independent_episode,FAULTS


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counts(rows,protocol):
    checked=[independent_episode(row,protocol) for row in rows]
    return dict(completed_episodes=len(rows),air=sum(x['airborne_contact'] for x in checked),
        conditional_recovery_count=sum(x['conditional_recovery'] for x in checked),
        complete=sum(x['complete'] for x in checked),numerical_failures=sum(x['numerical_failures'] for x in rows))


def main():
    assert not(HERE/'REPORT.json').exists()
    initial=json.loads((HERE/'INITIAL_INDEPENDENT_CHECK.json').read_text())
    raw=gzip.decompress((HERE/'all_closed_training_episodes.json.gz').read_bytes())
    assert hashlib.sha256(raw).hexdigest()==initial['source_json_sha256']
    rows=json.loads(raw);snapshot=json.loads((HERE/'progress.json').read_text())
    assert snapshot['update']==35 and len(rows)==15549
    protocol=json.loads((ROOT/'configs/g07_full_racket_eval_v4_r3.json').read_text());protocol['_root']=str(ROOT)
    history={}
    for line in (RUN/'stdout.log').read_text(errors='replace').splitlines():
        if not line.startswith('{'):continue
        try:item=json.loads(line)
        except json.JSONDecodeError:continue
        if all(key in item for key in ['update','actual_control_env_steps','completed_episodes','complete']):
            history[item['update']]=dict(line=line,record=item,line_sha256=hashlib.sha256(line.encode()).hexdigest())
    reconstructed=[]
    for update in [33,34,35]:
        entry=history[update]['record'];prefix=rows[:entry['completed_episodes']]
        checked=counts(prefix,protocol)
        assert all(entry[name]==value for name,value in checked.items())
        reconstructed.append(dict(update=update,independent_counts=checked,
            source='Reconstructed append-only prefix of preserved update35 rows, checked against the actual historical stdout progress line; not a captured old JSON file.',
            source_stdout_line_sha256=history[update]['line_sha256']))
    assert reconstructed[0]['independent_counts']['complete']==0
    assert reconstructed[1]['independent_counts']['complete']==1
    assert reconstructed[2]['independent_counts']['complete']==2
    elapsed={};details=[];iterations=set()
    for row in rows:
        lane=row['env_id'];start=elapsed.get(lane,0)
        lifetime=row['terminal_native_state']['time_s']
        duration=round(lifetime/.01)
        assert abs(duration-lifetime/.01)<.002 and 0<duration<=800
        end=start+duration;elapsed[lane]=end
        if not independent_episode(row,protocol)['complete']:continue
        first=INITIAL_ITERATION+start//128
        last=INITIAL_ITERATION+(end-1)//128
        contact=INITIAL_ITERATION+int((start+row['contact_time_s']/.01)//128)
        encountered=list(range(first,last+1));iterations.update(encountered)
        velocity=np.asarray(row['outgoing_velocity_mps'])
        details.append(dict(case_id=row['case_id'],env_id=lane,episode_index=row['episode_index'],
            source_training_index=row['source_training_index'],job_control_ticks=dict(start=start,end=end),
            local_episode_duration_s=lifetime,actor_iterations_encountered=encountered,
            contact_rollout_actor_iteration=contact,
            source_row_index=rows.index(row),
            contact_point_m=row['contact_point_m'],outgoing_velocity_mps=row['outgoing_velocity_mps'],
            outgoing_speed_mps=float(np.linalg.norm(velocity)),first_landing_m=row['first_landing_m'],
            crossing_point_m=row['crossing_point_m'],net_margin_m=row['net_margin_m'],
            shuttle_net_contact=row['shuttle_net_contact'],recovery_s=row['jump']['recovery_s'],
            maximum_body_ground_force_n=row['jump']['maximum_body_ground_force_n'],
            fault_flags=[name for name in FAULTS if row[name]],full_training_motion_trajectory_available=False))
    assert len(details)==2
    # This is the gradient update after the first full reward was observed;
    # its new weights did not execute that already-completed episode.
    iterations.add(34);iterations.add(35)
    lineage=[]
    for iteration in sorted(iterations):
        path=RUN/'ppo'/f'model_{iteration-1}.pt';data=path.read_bytes()
        state=torch.load(io.BytesIO(data),map_location='cpu',weights_only=False)
        assert state['iter']==iteration
        assert state['native']['total_timesteps']==iteration*2048*128
        assert not state['native']['canonical_restart']
        assert all(torch.isfinite(t).all() for t in state['model_state_dict'].values() if torch.is_tensor(t))
        assert all(torch.isfinite(t).all() for value in state['optimizer_state_dict']['state'].values()
            for t in value.values() if torch.is_tensor(t))
        lineage.append(dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(data).hexdigest(),
            iteration=iteration,cumulative_new_learner_control_env_steps=state['native']['total_timesteps'],
            model_and_Adam_finite=True,canonical_native_resume=False,
            role='Rollout-start actor state or post-completion update; physical diagnostic only. Normalization/exploration may evolve within each episode.'))
    first_end=details[0]['job_control_ticks']['end'];second_end=details[1]['job_control_ticks']['end']
    assert (34-1-INITIAL_ITERATION)*128<first_end<=(34-INITIAL_ITERATION)*128
    assert (35-1-INITIAL_ITERATION)*128<second_end<=(35-INITIAL_ITERATION)*128
    (HERE/'historical_progress_lines.json').write_text(json.dumps({str(k):history[k] for k in [33,34,35]},indent=2)+'\n')
    (HERE/'weight_lineage.json').write_text(json.dumps(lineage,indent=2)+'\n')
    report=dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='Two complete episodes in actual changing-policy joint-reward TRAIN, independently rescored. The first appeared in update34; the first preserved all-rows snapshot is update35. No frozen-policy or formal A/B performance claim.',
        source_run=str(RUN.relative_to(ROOT)),snapshot_update=35,
        first_complete_counter_update=34,first_preserved_snapshot_update=35,
        all_closed_training_counts=counts(rows,protocol),historical_prefix_checks=reconstructed,
        complete_episodes=details,source_control_start='Actual canonical ground restart at iteration8; each per-lane completed local duration is summed to recover its exact job control ticks.',
        lineage_note='Each episode crosses several actor updates and continuously evolving normalization/exploration. The contact actor and final actor are not interchangeable. No single saved diagnostic checkpoint is credited with a frozen complete success.',
        net_margin_note='Recorded signed center-minus-radius/top diagnostic at trailing-plane crossing, not minimum sphere/net clearance. Actual physical shuttle-net flags and original rules govern legality.',
        original_criteria_unchanged=True,final_contact_record_confirmation_pending=True,
        full_training_motion_recorded=False,video_note='No full training motion was saved; do not reconstruct a success video from endpoints. Prospectively retest original TRAIN65/6 in the already planned fixed40 assessment, retaining both actual outcomes.',
        source_sha256={str(path.relative_to(ROOT)):sha(path) for path in [RUN/'INPUTS.json',RUN/'source_snapshot.zip',
            RUN/'initial_ground_states.json',HERE/'complete_episodes.json',HERE/'all_closed_training_episodes.json.gz',
            Path(__file__).resolve()]},formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    (HERE/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(counts=report['all_closed_training_counts'],episodes=details,
        diagnostic_weights_checked=len(lineage),formal_acceptance=False)),flush=True)


if __name__=='__main__':main()
