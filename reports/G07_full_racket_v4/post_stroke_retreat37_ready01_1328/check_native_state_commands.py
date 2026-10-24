"""Compare post-stroke commands on saved failed CEM19states, not new dynamics."""
import argparse
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys
import numpy as np
import torch

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source-root',type=Path,required=True)
a=p.parse_args()
sys.path.insert(0,str(a.source_root.resolve()))
from quadsmash.control.full_racket_teacher_v4 import FullRacketTeacherV4

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
SNAP=OUT.parent/'early_landing37_native_tick400_snapshot'
torch.set_num_threads(2)
old=torch.load(OUT.parent/'early_landing37_ready01_1310/policy.pt',map_location='cpu',weights_only=True)
new=torch.load(OUT/'policy.pt',map_location='cpu',weights_only=True)
ids=[3,47,71,97]
state=torch.load(SNAP/'partial_search_state.pt',map_location='cpu',weights_only=False)
parameters=state['parameters'][np.array(ids)//2]
teachers=[FullRacketTeacherV4(x['backbone'],x['seed'],a.source_root/'configs/g07_go2_walk_transfer.json',len(ids),'cpu') for x in [old,new]]
for teacher in teachers:teacher.reset()
contract=json.loads((OUT.parent/'CPU_REPLAY_AUTHORED_JOINTS_0724.json').read_text())
names=[f'{leg}_{joint}_joint' for joint in ['hip','thigh','calf'] for leg in ['FL','FR','RL','RR']]+[f'joint{i}' for i in range(1,7)]
defaults=contract['defaults_before_neutral_arm_override']
defaults.update({f'joint{i+1}':q for i,q in enumerate(json.loads((ROOT/'configs/full_racket_reset_v4.json').read_text())['arm_joint_positions_rad'])})
default=torch.tensor([defaults[n] for n in names]);limits=torch.tensor([contract['limits'][n] for n in names])
times={};offsets={}
for log in json.loads((SNAP/'causal_plans.partial.json').read_text()):
    for i,env_id in enumerate(log['env_ids']):
        if env_id not in ids:continue
        assert log['valid'][i] and 2.3<=log['calibrated_event_time_s'][i]<=3.3
        times[env_id]=log['calibrated_event_time_s'][i]
        offsets[env_id]=np.array(log['position_m'][i])-np.array(old['seed']['desired_ball_center_m'])
event=torch.tensor([times[e] for e in ids],dtype=torch.float32)
offset=torch.tensor(np.array([offsets[e] for e in ids]),dtype=torch.float32)
commands=[];support=[];valids=[];clocks=[];masks=[]
with np.load(SNAP/'near_contact_all.partial.npz') as saved,torch.inference_mode():
    for index in range(len(saved['observations'])):
        obs=torch.tensor(saved['observations'][index,ids]);valid=torch.tensor(saved['valid_first_episode'][index,ids])
        post=obs[:,775]>=event-parameters[:,14]+parameters[:,19]
        values=[t.action({'policy':obs},parameters,default,limits,event_time=event,target_offset_world=offset) for t in teachers]
        assert torch.equal(values[0][:,:12],values[1][:,:12])
        assert torch.equal(values[0][valid&~post],values[1][valid&~post])
        assert all(torch.isfinite(x).all() and (abs(x)<=1).all() for x in values)
        q=[limits[:,0]+.5*(v+1)*(limits[:,1]-limits[:,0]) for v in values]
        # Complete physical tool support, without above-net exemptions. This
        # is command geometry, not an actual future robot or collision pose.
        x=[t.joint_preparation.support_state(j[:,12:],obs,t.kinematics)[0] for t,j in zip(teachers,q)]
        commands.append(torch.stack(values).numpy());support.append(torch.stack(x).numpy())
        valids.append(valid.numpy());clocks.append(obs[:,775].numpy());masks.append(post.numpy())
commands=np.stack(commands);support=np.stack(support);valids=np.stack(valids);clocks=np.stack(clocks);masks=np.stack(masks)
rows=[]
for j,e in enumerate(ids):
    selected=valids[:,j]&masks[:,j]&(clocks[:,j]>=2.98)
    assert selected.any()
    before,after=support[selected,0,j],support[selected,1,j]
    rows.append(dict(env_id=e,valid_post_stroke_samples=int(selected.sum()),
        old_command_maximum_tool_x_m=float(before.max()),new_command_maximum_tool_x_m=float(after.max()),
        median_command_tool_x_reduction_m=float(np.median(before-after)),
        samples_old_full_tool_beyond_net=int((before>1.5).sum()),samples_new_full_tool_beyond_net=int((after>1.5).sum()),
        samples_new_tool_within1p151_m=int((after<=1.151).sum())))
assert np.any(commands[:,0,:,12:]!=commands[:,1,:,12:])
np.savez_compressed(OUT/'native_state_counterfactual_commands.npz',actions=commands,commanded_full_tool_x_m=support,
                    valid_first_episode=valids,clock_s=clocks,post_stroke=masks,env_ids=ids)
paths=[SNAP/'near_contact_all.partial.npz',SNAP/'partial_search_state.pt',SNAP/'causal_plans.partial.json',OUT/'policy.pt',Path(__file__),a.source_root/'quadsmash/control/full_racket_joint_preparation_v4.py',a.source_root/'quadsmash/control/full_racket_teacher_v4.py']
report=dict(created_at=datetime.now(timezone.utc).isoformat(),
    scope='Counterfactual commands on four saved failed CEM19native public streams and original causal forecasts. Not changed-state dynamics, collision avoidance or recovery proof.',
    env_ids=ids,ticks=len(commands),all_commands_finite_bounded=True,leg_commands_bitwise_equal_on_identical_inputs=True,
    pre_stroke_commands_bitwise_equal=True,post_stroke_command_comparison=rows,native_execution=False,formal_acceptance=False,
    source_sha256={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
(OUT/'NATIVE_STATE_COMMAND_CHECK.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))
