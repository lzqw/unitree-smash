"""Independently assess the completed first origin0 reference candidate."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from quadsmash.control.full_racket_prepose_branch_v4 import PreposeBranchControllerV4
from quadsmash.control.full_racket_body_search_v4 import summarize_candidate
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (HERE/'RESULT.json').exists()
    run=ROOT/'reports/G07_full_racket_v4/prepose_origin0_8_sequential_n128_01'
    inputs=read(run/'INPUTS.json');entry=read(run/'metrics.json')['history'][0]
    assert entry['candidate_id']==0 and entry['candidate']['overrides']=={} and entry['control_ticks']==800
    rows=read(run/'episodes_000.json');initial=read(run/'initial_ground_states_000.json')
    protocol=read(ROOT/'configs/g07_full_racket_eval_v4_r3_origin0.json');protocol['_root']=str(ROOT)
    cases,indices=training_cases(ROOT,protocol)
    assert len(rows)==128 and indices==list(range(128)) and initial['all_clocks_zero']
    assert np.asarray(initial['actual_environment_origins_m']).shape==(128,3) and not np.asarray(initial['actual_environment_origins_m']).any()
    for lane,row in enumerate(rows):
        assert row['env_id']==row['jump']['env_id']==lane and row['source_training_index']==lane
        assert row['case_id']==cases[lane]['case_id'] and row['episode_index']==initial['episode_ids'][lane]
        assert row['ground_init'] and row['candidate_id']==0
    summary=summarize_candidate(rows,protocol);assert summary==entry['summary'] and summary['numerical_failures']==0
    source_path=Path(read(ROOT/'reports/G07_full_racket_v4/prepose_origin0_after_precision_queue02/ACTUAL_PROPOSALS.json')['source_policy'])
    assert sha(source_path)==inputs['source_policy_sha256']
    source=torch.load(source_path,map_location='cpu',weights_only=False)
    default,limits=[torch.tensor(inputs[key],dtype=torch.float32) for key in ['default_joint_position_rad','joint_limits_rad']]
    with np.load(run/'public_and_actions_000.npz') as arrays:
        public,actions,valid=[arrays[key].copy() for key in ['observations','actions','valid_first_episode']]
    controller=PreposeBranchControllerV4(source,{},2,'cpu');maximum=0.;previous=None
    with torch.inference_mode():
        for observation,actual,keep in zip(public,actions,valid):
            current=torch.from_numpy(observation)
            if previous is not None:
                reset=torch.where(current[:,775]<previous-1e-5)[0]
                if len(reset):controller.reset(reset)
            command=controller.action({'policy':current},default,limits).numpy()
            if keep.any():maximum=max(maximum,float(abs(command[keep]-actual[keep]).max()))
            previous=current[:,775].clone()
    assert maximum<=5e-4
    poses=run/'all_first_episodes_000.npz'
    with np.load(poses) as arrays:
        assert np.isfinite(arrays['link_poses']).all() and np.isfinite(arrays['ball_position_m']).all()
    sources=[run/name for name in ['episodes_000.json','contacts_000.json','all_first_episodes_000.npz',
        'public_and_actions_000.npz','initial_ground_states_000.json','causal_plans_000.json','INPUTS.json']]
    values=dict(checked_at=datetime.now(timezone.utc).isoformat(),summary=summary,
        actual_completed_reference_episodes=128,actual_reference_control_env_steps=102400,
        actual_valid_public_rows=int(valid.sum()),CPU_vs_actual_GPU_max_action_error=maximum,original_action_tolerance=5e-4,
        actual_racket_contacts=sum(row['had_contact'] for row in rows),actual_frame_contacts=sum(row['frame_hit'] for row in rows),
        actual_string_hits=sum(row['string_hit'] for row in rows),qualified_flight_seen=sum(row['jump']['qualified_flight_seen'] for row in rows),
        body_landing_count=sum(row['jump']['body_landing'] for row in rows),stable_recovery_count=sum(row['jump']['stable_recovery'] for row in rows),
        actual_precontact_and_recovery_failures_retained=True,source_sha256={str(path):sha(path) for path in sources},
        source_weights_and_commands_matched=True,source_old_layout_performance_not_inherited=True,
        scope='Completed reference candidate only; overall eight-candidate job remains running. No new cold or formal score inferred.',
        formal_cases_read=False,formal_acceptance=False)
    (HERE/'ACTUAL_REFERENCE_ENTRY.json').write_text(json.dumps(entry,indent=2)+'\n')
    (HERE/'RESULT.json').write_text(json.dumps(values,indent=2)+'\n');print(json.dumps(values),flush=True)


if __name__=='__main__':main()
