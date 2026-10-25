"""Compare actual frozen8/40 inputs, base policy and precontact full-body motion."""
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
REPORTS=ROOT/'reports/G07_full_racket_v4'


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def differences(left,right,prefix=''):
    if torch.is_tensor(left):return []if torch.is_tensor(right)and torch.equal(left,right)else[prefix]
    if type(left)!=type(right):return[prefix]
    if isinstance(left,dict):
        if set(left)!=set(right):return[prefix+'.keys']
        return [item for key in left for item in differences(left[key],right[key],prefix+'.'+str(key))]
    if isinstance(left,(list,tuple)):
        if len(left)!=len(right):return[prefix+'.length']
        return [item for i in range(len(left))for item in differences(left[i],right[i],prefix+f'[{i}]')]
    if isinstance(left,float)and math.isnan(left)and math.isnan(right):return[]
    return []if left==right else[prefix]


def main():
    output=HERE/'ACTUAL_MOTION_COMPARISON.json';assert not output.exists()
    runs=[REPORTS/'residual_ppo_joint_placement_actor32_n2048_probe8_fixed_train128_01',
        REPORTS/'residual_ppo_joint_placement_actor32_n2048_iter40_fixed_train128_01']
    policies=[REPORTS/'residual_ppo_joint_placement_actor32_n2048_probe8_01/policy.pt',
        REPORTS/'residual_ppo_joint_placement_actor32_n2048_resume32_01/policy.pt']
    base_path=REPORTS/'cem32_lateral_feedback_1015/best_candidate.pt'
    base=torch.load(base_path,map_location='cpu',weights_only=False);dependency_paths=[base_path,Path(__file__).resolve()]
    snapshots=[];base_diffs=[];source_maps=[];summaries=[]
    for run,policy_path in zip(runs,policies):
        assert read(run/'job.json')['status']=='SUCCEEDED'
        policy=torch.load(policy_path,map_location='cpu',weights_only=False)
        diff=differences(base,policy['base_controller']);assert not diff,diff
        base_diffs.append(diff)
        geometry=read(run/'geometry.json');poses=np.load(run/'all_first_episodes.npz')
        snapshots.append(dict(names=geometry['body_names'],poses=poses['link_poses'],
            ball=poses['ball_position_m'],clock=poses['episode_clock_s'],valid=poses['valid_first_episode']))
        source_maps.append(read(run/'source_hashes.json'))
        metrics=read(run/'metrics.json');summaries.append({k:metrics[k]for k in ['complete','airborne_contacts','conditional_recovery_count','numerical_failures','actual_racket_contacts']})
        dependency_paths.extend([policy_path,run/'job.json',run/'metrics.json',run/'source_hashes.json',
            run/'all_first_episodes.npz',run/'initial_states.json',run/'public_and_actions_first2.npz'])
    before,after=snapshots;assert before['names']==after['names']
    names=before['names'];assert names[0]=='base';tool=names.index('Racket')
    keys={k for k in source_maps[0]if k.startswith('quadsmash/')or k=='scripts/evaluate_full_racket_train_v4.py'}
    assert keys and all(source_maps[0][k]==source_maps[1][k]for k in keys)
    observations=[np.load(run/'public_and_actions_first2.npz')['observations']for run in runs]
    initial_public_equal=np.array_equal(observations[0][0],observations[1][0])
    selected=(before['valid']&after['valid']&(before['clock']>=2.)&(before['clock']<=2.5))
    assert selected.any()
    ball_delta=np.linalg.norm(before['ball']-after['ball'],axis=-1)
    precontact_ball_error=float(ball_delta[selected].max())
    measurements=[]
    for seconds in [0.,1.,2.,2.4,2.6,2.7]:
        frame=round(seconds/.02);valid=before['valid'][frame]&after['valid'][frame]
        assert valid.any()
        a,b=before['poses'][frame,valid],after['poses'][frame,valid]
        root_delta=np.linalg.norm(a[:,0,:3]-b[:,0,:3],axis=-1)
        tool_delta=np.linalg.norm(a[:,tool,:3]-b[:,tool,:3],axis=-1)
        qa=Rotation.from_quat(a[:,0,3:][:,[1,2,3,0]])
        qb=Rotation.from_quat(b[:,0,3:][:,[1,2,3,0]])
        angle=(qa.inv()*qb).magnitude()*180/np.pi
        measurements.append(dict(time_s=seconds,matched_valid_cases=int(valid.sum()),
            root_displacement_median_m=float(np.median(root_delta)),root_displacement_max_m=float(root_delta.max()),
            root_rotation_difference_median_deg=float(np.median(angle)),
            string_center_displacement_median_m=float(np.median(tool_delta)),string_center_displacement_max_m=float(tool_delta.max()),
            original8_strings_height_median_m=float(np.median(a[:,tool,2])),new40_strings_height_median_m=float(np.median(b[:,tool,2]))))
    initial_diffs=differences(read(runs[0]/'initial_states.json'),read(runs[1]/'initial_states.json'))
    result=dict(checked_at=datetime.now(timezone.utc).isoformat(),
        scope='Matched actual native frozen TRAIN128 trajectories for new learner8 and40. No counterfactual simulation or isolated causal ablation.',
        frozen_summaries=summaries,base_controller_exactly_equal_to_original_for_both=not any(base_diffs),
        unchanged_common_environment_control_and_evaluator_source_files=len(keys),
        initial_state_artifacts_exactly_equal=not initial_diffs,initial_state_different_fields=initial_diffs,
        first_saved_public_observations_bitwise_equal=initial_public_equal,
        measured_precontact_ball_samples=int(selected.sum()),precontact_ball_position_max_difference_m=precontact_ball_error,
        precontact_body_and_strings_comparison=measurements,
        interpretation='Both policy exports retain the exact original analytic base controller. Compare measured closed-loop body/tool drift alongside unchanged precontact incoming ball positions. Coupled policy differences are not an isolated causal explanation, and these data do not establish which reward/control change will fix the collapse.',
        source_sha256={str(p.relative_to(ROOT)):sha(p)for p in dependency_paths},
        formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='source_sha256'}),flush=True)


if __name__=='__main__':main()
