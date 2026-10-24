"""Compare two actually executed, paired native prefixes; never infer recovery."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2];BASE=OUT.parent
sys.path.insert(0,str(ROOT))
from quadsmash.control.full_racket_joint_preparation_v4 import FullRacketJointPreparation
torch.set_num_threads(2)
guard=FullRacketJointPreparation({})
ids=[3,47,71,97]
snapshots=[BASE/'early_landing37_native_tick400_snapshot',BASE/'post_stroke_retreat37_native_tick400_snapshot']
labels=['CEM19: earlier leg targets','CEM20: added post-stroke retreat']
colors=['#d97918','#1569ae']

def first_rows(snapshot):
    rows={}
    for row in json.loads((snapshot/'episodes.partial.json').read_text()):rows.setdefault(row['env_id'],row)
    return rows

def support_x(poses):
    poses=np.asarray(poses)
    direction=Rotation.from_quat(poses[:,[4,5,6,3]]).inv().apply(np.tile([1.,0,0],(len(poses),1)))
    with torch.inference_mode():local=guard.local_support(torch.tensor(direction)).numpy()
    return poses[:,0]+(direction*local).sum(1)

data=[np.load(s/'near_contact_all.partial.npz') for s in snapshots]
rows=[first_rows(s) for s in snapshots]
curves=[];case_reports=[]
for e in ids:
    case=dict(env_id=e,native_executions=[])
    for index,(snapshot,a,first) in enumerate(zip(snapshots,data,rows)):
        valid=a['valid_first_episode'][:,e]&(a['episode_clock_s'][:,e]>=2.72)
        times=a['episode_clock_s'][valid,e].astype(float)
        poses=a['link_poses'][valid,e,27].astype(float)
        root=a['link_poses'][valid,e,0,2].astype(float)
        row=first.get(e)
        if row is not None:
            terminal=row['terminal_native_state'];clock=terminal['time_s']
            if times[-1]<clock:
                times=np.r_[times,clock]
                poses=np.r_[poses,np.array([terminal['body_link_positions_m'][27]+terminal['body_link_quaternions_wxyz'][27]])]
                root=np.r_[root,terminal['body_link_positions_m'][0][2]]
        x=support_x(poses)
        curves.append(dict(env_id=e,variant=index,time_s=times,actual_full_tool_maximum_x_m=x,actual_root_height_m=root))
        case['native_executions'].append(dict(label=labels[index],samples=len(times),last_clock_s=float(times[-1]),
            maximum_actual_full_tool_x_m=float(x.max()),minimum_observed_root_height_m=float(root.min()),
            final_label_available=row is not None,
            robot_net_fault=None if row is None else row['robot_net_fault'],
            body_ground_terminal=None if row is None else row['training_terminal_reasons']['body_ground'],
            recovery_success=None if row is None else row['recovery_success']))
    case_reports.append(case)

fig,axes=plt.subplots(2,4,figsize=(15,7),sharex=True,sharey='row')
for col,e in enumerate(ids):
    for curve in [c for c in curves if c['env_id']==e]:
        index=curve['variant'];axes[0,col].plot(curve['time_s'],curve['actual_full_tool_maximum_x_m'],color=colors[index],label=labels[index])
        axes[1,col].plot(curve['time_s'],curve['actual_root_height_m'],color=colors[index])
    axes[0,col].axhline(1.5,color='#777777',linestyle='--',label='Net plane')
    axes[0,col].set_title(f'Previously selected env {e}')
    axes[1,col].set_xlabel('Native clock (s)')
    for row in range(2):axes[row,col].grid(alpha=.2)
axes[0,0].set_ylabel('Actual full-tool max x (m)');axes[1,0].set_ylabel('Actual root height (m)')
fig.suptitle('Actual paired native executions: retreat reduces net approach, recovery remains unproven')
handles,names=axes[0,0].get_legend_handles_labels();fig.legend(handles,names,loc='lower center',ncol=3)
fig.tight_layout(rect=(0,.07,1,.94));figure=OUT/'PAIRED_ACTUAL_RETREAT.png';fig.savefig(figure,dpi=150);plt.close(fig)

old_pop=torch.load(BASE/'ball_early_landing37_cem19/population_000.pt',map_location='cpu',weights_only=False)['parameters']
new_pop=torch.load(snapshots[1]/'partial_search_state.pt',map_location='cpu',weights_only=False)['parameters']
assert torch.equal(old_pop,new_pop)
contacts=[]
for s in snapshots:contacts.append({(c['env_id'],c['event_index']):c for c in json.loads((s/'contacts.partial.json').read_text()) if c['episode_index']==0})
assert contacts[0].keys()==contacts[1].keys()
fields=['time_s','point_m','string_contact','frame_contact','point_velocity_mps','ball_velocity_mps']
assert all(all(contacts[0][k][f]==contacts[1][k][f] for f in fields) for k in contacts[0])
keys=['airborne_string_hit','crossed_net','landed_legal_court','landed_target','robot_net_fault','recovery_success','complete_smash']
counts=[]
for first in rows:
    counts.append(dict(closed=len(first),pending=128-len(first),**{k:sum(bool(r[k]) for r in first.values()) for k in keys},
                       body_ground_terminal=sum(r['training_terminal_reasons']['body_ground'] for r in first.values())))
sources=[Path(__file__),ROOT/'quadsmash/control/full_racket_joint_preparation_v4.py',ROOT/'configs/full_racket_v4_r3.json']
for s in snapshots:sources.extend(s/n for n in ['near_contact_all.partial.npz','episodes.partial.json','contacts.partial.json','partial_search_state.pt'])
report=dict(created_at=datetime.now(timezone.utc).isoformat(),
    scope='Paired actually executed first400ticks with identical64x37population,80exact matching contact events and previously verified identical first100native prefix. Whole-generation labels remain pending; formal A200+B200episodes are unrun.',
    population_bitwise_equal=True,exact_matching_contact_events=len(contacts[0]),contact_comparison_fields=fields,
    variants=[dict(label=label,counts_at400ticks=count) for label,count in zip(labels,counts)],cases=case_reports,
    figure=figure.name,figure_sha256=hashlib.sha256(figure.read_bytes()).hexdigest(),
    physical_support_method='Analytic support of all97authored racket/fixture primitives transformed by recorded actual native Racket body poses; terminal snapshots appended when available. No commanded/counterfactual pose is plotted.',
    limitations=['Above-net followthrough may legally crossx1.5; actual rule flags, not x alone, determine faults.',
                 'Earlier old net terminations censor later old body outcomes; different body-ground counts are not a direct estimate of a fall-rate effect.',
                 'Four cases were selected for command diagnostics before the new native outcome; they are not panel statistics.',
                 'Pending episodes do not receive inferred recovery or success labels.'],
    native_execution=True,formal_acceptance=False,
    sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
(OUT/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
np.savez_compressed(OUT/'paired_actual_curves.npz',**{f'env{c["env_id"]}_variant{c["variant"]}_{k}':v for c in curves for k,v in c.items() if k not in ['env_id','variant']})
print(json.dumps({k:v for k,v in report.items() if k not in ['sha256','cases','limitations']},indent=2))
