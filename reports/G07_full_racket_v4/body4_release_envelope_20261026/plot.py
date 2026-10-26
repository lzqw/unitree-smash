"""Show measured body4 release positions and limited CPU direction witnesses."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
rows=json.loads((HERE/'RESULT.json').read_text())['proposals']
positions=np.asarray([r['actual_ball_position_m'] for r in rows])
found=np.asarray([bool(r['sampled_feasible_pitch_count']) for r in rows])
fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
for mask,color,label in [(found,'#287d46','Sampled direction witness (29)'),(~found,'#bd651b','No witness in this grid (30)')]:
    axes[0].scatter(positions[mask,0],positions[mask,2],c=color,s=30,label=label)
    axes[1].scatter(positions[mask,1],positions[mask,2],c=color,s=30,label=label)
complete=next(i for i,r in enumerate(rows) if r['TRAIN_index']==36)
for ax,x in zip(axes,[positions[:,0],positions[:,1]]):
    ax.scatter(x[complete],positions[complete,2],c='black',s=130,marker='*',label='Actual complete TRAIN36')
    ax.grid(alpha=.25);ax.set_ylabel('Measured ball center height (m)')
axes[0].set_xlabel('Measured ball center X (m)')
axes[1].set_xlabel('Measured ball center Y (m)')
axes[0].legend(fontsize=8,loc='lower right')
fig.suptitle('Body4: 59 actual airborne contacts, one actual complete episode out of128')
fig.supxlabel('CPU direction grid only; no robot reach/impact/recovery proof. Grid failure is not an impossibility bound.',fontsize=9)
fig.savefig(HERE/'Actual_body4_release_positions_and_direction_screen_20261026.png',dpi=180)
