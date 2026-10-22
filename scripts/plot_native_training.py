#!/usr/bin/env python3
"""Plot measured native PPO progress, without converting reward to success."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('runs',nargs='+',type=Path)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for path in a.runs:
    rows=[json.loads(line) for line in (path/'native/training_progress.jsonl').read_text().splitlines() if line.strip()]
    x=[r['train_env_steps_this_run']/1e6 for r in rows]
    for ax,key,label in zip(axes.flat,
        ('recent_success_rate','recent_qualified_jump_rate','recent_mean_com_rise_m','recent_body_landing_rate'),
        ('Complete jump + recovery fraction','Qualified jump landing fraction','Mean total COM rise (m)','Non-foot ground contact fraction')):
        ax.plot(x,[r[key] for r in rows],label=path.name)
        ax.set(xlabel='Measured training interactions in this run (millions)',ylabel=label)
        ax.grid(alpha=.25)
axes[0,0].set_ylim(-.02,1.02);axes[0,1].set_ylim(-.02,1.02);axes[1,1].set_ylim(-.02,1.02)
axes[0,0].legend(fontsize=8)
fig.suptitle('Native PPO training samples: rolling last 2,048 episodes; not independent evaluation')
a.output.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(a.output,dpi=160)
print(a.output)
