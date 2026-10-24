"""Plot reconstructed actual tool support against two counterfactual commands."""
import argparse
from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-root',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(a.source_root))
from quadsmash.control.full_racket_kinematics_v4 import FullRacketKinematicsV4
from quadsmash.control.full_racket_joint_preparation_v4 import FullRacketJointPreparation
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
torch.set_num_threads(2)
replay=np.load(OUT/'native_state_counterfactual_commands.npz')
native=np.load(OUT.parent/'early_landing37_native_tick400_snapshot/near_contact_all.partial.npz')
ids=replay['env_ids'];q0=np.array(json.loads((ROOT/'configs/full_racket_reset_v4.json').read_text())['arm_joint_positions_rad'])
model=FullRacketKinematicsV4().float();guard=FullRacketJointPreparation({}).float()
actual=[]
with torch.inference_mode():
    for obs in native['observations'][:,ids]:
        public=torch.tensor(obs);q=torch.tensor(q0+obs[:,21:27],dtype=torch.float32)
        actual.append(guard.support_state(q,public,model,observed_center=True)[0].numpy())
actual=np.stack(actual)
fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True)
for j,axis in enumerate(axes.flat):
    valid=replay['valid_first_episode'][:,j]&(replay['clock_s'][:,j]>=2.72)
    t=replay['clock_s'][valid,j]
    axis.plot(t,actual[valid,j],color='black',label='Reconstructed actual tool envelope')
    axis.plot(t,replay['commanded_full_tool_x_m'][valid,0,j],color='#d97915',label='Old command geometry')
    axis.plot(t,replay['commanded_full_tool_x_m'][valid,1,j],color='#176cb0',label='Proposed retreat command geometry')
    axis.axhline(1.5,color='#999999',linestyle='--',label='Net plane')
    axis.set_title(f'Native CEM19 env {ids[j]} (failed episode)');axis.grid(alpha=.2)
    axis.set_xlabel('Native episode clock (s)');axis.set_ylabel('Full tool maximum world x (m)')
fig.suptitle('Post-stroke tracking gap: proposed commands have not been executed')
handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=9)
fig.tight_layout(rect=(0,.10,1,.95));dest=OUT/'POST_STROKE_TRACKING_GAP.png';fig.savefig(dest,dpi=160);plt.close(fig)
report=dict(scope='Tool envelope reconstructed from actual native public joint states/observed tool center. Proposed trajectory is command geometry only, not actual tracking or collision avoidance. Above-net followthrough may legally crossx1.5; original rules still decide faults.',
            env_ids=ids.tolist(),ticks=len(actual),figure=str(dest.relative_to(ROOT)),figure_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
            numeric_source_sha256=hashlib.sha256((OUT/'native_state_counterfactual_commands.npz').read_bytes()).hexdigest(),
            native_execution_of_proposal=False,formal_acceptance=False)
(OUT/'TRACKING_GAP_PLOT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
