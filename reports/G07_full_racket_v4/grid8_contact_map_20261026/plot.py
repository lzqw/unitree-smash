"""Plot measured contacts on the frozen string geometry and outcome funnel."""
import csv,hashlib,json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
MAIN=ROOT.parent/'unitree_smash'
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(MAIN))
from quadsmash.physics.full_racket_v4 import string_polygon,string_strips

config_path=MAIN/'configs/full_racket_v4_r3.json'
config=json.loads(config_path.read_text())
rows=list(csv.DictReader((HERE/'CONTACT_MAP.csv').open()))
policy_path=MAIN/'reports/G07_full_racket_v4/body_strike_grid16_n2048_01/best_candidate.pt'
policy=torch.load(policy_path,map_location='cpu',weights_only=False)
aim=np.asarray(policy['source_public_policy']['seed']['string_uv_m'])*100
fig,(ax,bar)=plt.subplots(1,2,figsize=(12,5.5),layout='constrained')
for strip in string_strips(config):
    center,size=np.asarray(strip['center'])*100,np.asarray(strip['size'])*100
    ax.add_patch(Rectangle((center[1]-size[1]/2,center[2]-size[2]/2),size[1],size[2],color='#e3edf3',linewidth=0))
polygon=string_polygon(config)*100;polygon=np.r_[polygon,polygon[:1]]
ax.plot(polygon[:,0],polygon[:,1],color='#346985',linewidth=1,label='Frozen strings boundary')
for flag,color,label in [('True','#287d46','Actual strings (55)'),('False','#bd651b','Actual frame only (35)')]:
    group=[r for r in rows if r['actual_string_hit']==flag]
    ax.scatter([float(r['contact_y_m'])*100 for r in group],[float(r['contact_z_m'])*100 for r in group],
        c=color,s=22,alpha=.85,label=label)
for index in [13,29]:
    row=next(r for r in rows if int(r['TRAIN'])==index)
    ax.scatter(float(row['contact_y_m'])*100,float(row['contact_z_m'])*100,marker='*',s=110,color='black',
        label='Actual complete (2)' if index==13 else None)
ax.scatter(*aim,marker='x',s=70,color='#5360a6',label='Measured teacher aim UV')
ax.axvline(0,color='gray',linewidth=.5);ax.axhline(0,color='gray',linewidth=.5)
ax.set(xlabel='Racket-frame lateral Y (cm)',ylabel='Racket-frame longitudinal Z (cm)',xlim=(-12,12),ylim=(-14,15))
ax.set_aspect('equal');ax.grid(alpha=.2);ax.legend(fontsize=7,loc='lower left')
labels=['All cases','Legal air\nstrings','Attack','Valid\ncross','Target','Stable\ncomplete']
values=[128,55,19,16,4,2]
bars=bar.bar(np.arange(6),values,color=['#728ba1']*5+['#287d46'])
bar.bar_label(bars,padding=3);bar.set_xticks(np.arange(6),labels,fontsize=8)
bar.set(ylabel='Cases in successive full-task conditions',ylim=(0,145));bar.grid(axis='y',alpha=.2)
bar.text(.03,.93,'38 misses and35 frame-only hits leave55 legal air strings.',transform=bar.transAxes,fontsize=8)
fig.suptitle('Fixed grid8 teacher: first actual contact location and all128 outcome funnel')
fig.supxlabel('Contact colors are native labels; this projected plot is not a new hit classifier or a causal experiment.',fontsize=9)
output=HERE/'Actual_grid8_contact_map_and_failure_funnel_20261026.png';fig.savefig(output,dpi=180)
paths=[Path(__file__),HERE/'CONTACT_MAP.csv',HERE/'RESULT.json',config_path,policy_path,output]
(HERE/'PLOT_RECEIPT.json').write_text(json.dumps(dict(source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},teacher_aim_uv_m=(aim/100).tolist(),new_native_execution=False,formal_acceptance=False),indent=2)+'\n')
