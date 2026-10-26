"""Plot measured funnel, face/outgoing errors and the CPU origin-rounding fit."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
result=json.loads((HERE/'RESULT.json').read_text())
rounding=json.loads((HERE/'FLOAT32_ORIGIN_COMPARISON.json').read_text())
out=HERE/'MEASURED_DIRECTION_AND_ORIGIN_ERRORS.png';assert not out.exists()
fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
f=result['funnel'];keys=['all_original_cases','air','attack','attack_legal_cross','attack_legal_cross_target','complete']
labels=['All128','Air','Attack','Legal cross','Target','Complete']
values=[f[key] for key in keys];bars=axes[0,0].bar(labels,values,color=['#94a3b8','#4c78a8','#5aa9c6','#55a686','#84b547','#29814a'])
axes[0,0].bar_label(bars,padding=3);axes[0,0].set_ylim(0,142);axes[0,0].set_ylabel('Actual native first episodes')
axes[0,0].set_title('Original TRAIN128 funnel: 5 complete; no formal result')
attack=[item for item in result['measurements'] if item['attack'] and item['first_landing_m'] is not None]
for item in attack:
    color='#198754' if item['complete'] else ('#d48a21' if item['attack_legal_cross'] else '#b94040')
    marker='x' if item['shuttle_net_contact'] else 'o'
    axes[0,1].scatter(item['actual_face_pitch_deg'],item['first_landing_m'][0],c=color,marker=marker,s=48)
    axes[1,0].scatter(item['actual_face_yaw_deg'],item['actual_outgoing_velocity_mps'][1],c=color,marker=marker,s=48)
axes[0,1].axhspan(2,4,color='#198754',alpha=.08)
desired=np.rad2deg(np.arctan2(result['source_requested_normal_world'][2],result['source_requested_normal_world'][0]))
axes[0,1].axvline(desired,color='#555',ls='--',label='Requested face pitch')
axes[0,1].set(xlabel='Actual contact face pitch (degrees)',ylabel='Actual first landing x (m)',title='Face tracking and landing (x markers: net touched)')
axes[0,1].legend(fontsize=8)
axes[1,0].axvline(0,color='#555',ls='--');axes[1,0].axhline(0,color='#555',ls=':')
axes[1,0].set(xlabel='Actual contact face yaw (degrees)',ylabel='Actual outgoing ball vy (m/s)',title='Observed sideways face and outgoing motion')
fields=['ideal_float64_max_error_m','float32_local_Euler_max_error_m','float32_world_Euler_max_error_m']
for key,label,color in zip(fields,['Float64 local ODE','Float32 local Euler','Float32 world Euler'],['#b94040','#d48a21','#198754']):
    y=np.maximum([item[key]*1000 for item in rounding['results']],1e-9)
    x=[item['TRAIN_index'] for item in rounding['results']]
    axes[1,1].scatter(x,y,label=label,color=color,s=17,alpha=.8)
axes[1,1].set_yscale('log');axes[1,1].set(xlabel='TRAIN case index',ylabel='Maximum difference from native (mm)',title='Rounding surrogate matches recorded world-position drift')
axes[1,1].legend(fontsize=8)
for ax in axes.flat:ax.grid(alpha=.18)
fig.suptitle('Measured Aim [0, 0.06]: failure mechanisms and numerical-origin diagnosis',fontsize=13)
fig.savefig(out,dpi=160);plt.close(fig)
csv_path=HERE/'ALL128_DIAGNOSTIC.csv';assert not csv_path.exists()
fields=['TRAIN_index','actual_native_episode_index','airborne_contact','attack','attack_legal_cross',
        'attack_legal_cross_target','conditional_recovery','complete','body_landing','shuttle_net_contact',
        'actual_face_pitch_deg','actual_face_yaw_deg','desired_vs_actual_face_angle_deg',
        'maximum_precontact_forecast_position_error_m','precontact_forecast_within_diagnostic_bound']
with csv_path.open('w') as stream:
    writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
    for item in result['measurements']:writer.writerow({key:item.get(key) for key in fields})
print(out,flush=True)
