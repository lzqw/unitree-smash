"""Compare actual50Hz root heights for the same TRAIN0/1 cases."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
with np.load(HERE/'actual_selected_roots.npz') as data:grid={k:data[k].copy() for k in data.files}
colds=[]
for i in [0,1]:
    with np.load(HERE/f'actual_cold_reference{i}_roots.npz') as data:colds.append({k:data[k].copy() for k in data.files})
fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained');samples=[]
for case,ax in enumerate(axes):
    streams=[('Prior cold body4',colds[0],case,'#186ca4'),('Exact body4 in grid',grid,case,'#c17b17'),
        ('Prior cold body6',colds[1],case,'#3f8b4e'),('Exact body6 in grid',grid,2+case,'#b74b4b'),
        ('Selected grid candidate8',grid,10+case,'#8752a6')]
    for label,data,column,color in streams:
        keep=data['valid'][:,column] & (data['clocks'][:,column]<=4.2)
        t=data['clocks'][keep,column];z=data['roots'][keep,column,2]
        ax.plot(t,z,label=label,color=color,linewidth=1.5)
        points=[]
        for target in [.5,1.,1.5,2.,2.5,3.]:
            if len(t) and target<=t[-1]+.001:
                i=int(abs(t-target).argmin());points.append(dict(time_s=float(t[i]),root_link_height_m=float(z[i])))
        samples.append(dict(TRAIN=case,source=label,actual_samples=points))
    ax.axvline(2.,color='gray',linestyle='--',linewidth=1,label='Active ball launch')
    ax.set(xlabel='Actual episode clock (s)',ylabel='Root link Z (m)',title=f'Same actual TRAIN{case}',ylim=(0,.82))
    ax.grid(alpha=.25)
axes[0].legend(fontsize=8,loc='upper right')
fig.suptitle('Identical controller parameters and ground reset do not reproduce the same native path')
fig.supxlabel('Measured50Hz samples; different batch size, seed and instances. Cause not established.',fontsize=9)
fig.savefig(HERE/'Actual_grid16_reference_root_heights_20261026.png',dpi=180)
(HERE/'ACTUAL_ROOT_HEIGHT_SAMPLES.json').write_text(json.dumps(samples,indent=2)+'\n')
