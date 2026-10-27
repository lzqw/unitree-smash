"""Render actual posture/support observations, without generating any motion."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    posture=json.loads((HERE/'ACTUAL_POSTURE_CHECK.json').read_text())
    stability=json.loads((HERE/'PUBLIC_STABILITY_CHECK.json').read_text())
    paths=[HERE/'POSTURE_PROFILES.npz',HERE/'PUBLIC_STABILITY_PROFILES.npz']
    assert sha(paths[0])==posture['profiles_sha256'] and sha(paths[1])==stability['profiles_sha256']
    with np.load(paths[0]) as d:pose={k:d[k].copy() for k in d.files}
    with np.load(paths[1]) as d:public={k:d[k].copy() for k in d.files}
    fig,axes=plt.subplots(3,2,figsize=(14,11),constrained_layout=True)
    candidates=[x['source_training_index'] for x in posture['runs'][1]['samples']]
    assert len(candidates)==13
    for ax,field,title in [(axes[0,0],'root_tilt_deg','13 actual air contacts: body tilt'),
                            (axes[0,1],'root_height_m','13 actual air contacts: root height')]:
        for lane in candidates:
            key=f'candidate1_TRAIN{lane}'
            t=pose[key+'_time_relative_landing_s'];v=pose[key+'_'+field]
            keep=(t>=-.2)&(t<=.6)
            ax.plot(t[keep],v[keep],color='#3f80be',alpha=.32,lw=1.2)
        for key,color,label in [('candidate0_TRAIN69','#bd3c36','Original source69: failed'),
                                ('recovered69','#16863c','Exploration69: recovered')]:
            t=pose[key+'_time_relative_landing_s'];v=pose[key+'_'+field];keep=(t>=-.2)&(t<=.6)
            ax.plot(t[keep],v[keep],color=color,lw=2.2,label=label)
        ax.set_title(title);ax.set_xlim(-.2,.6)
        ax.set_ylabel('deg' if field.endswith('deg') else 'm')
    axes[0,0].legend(fontsize=8,loc='upper left')
    comparisons=[('candidate0_TRAIN69','#bd3c36','Original0'),('candidate1_TRAIN69','#3f80be','Early gain0.5'),
                 ('candidate2_TRAIN69','#9a69b1','Early gain1'),('recovered69','#16863c','Recovered exploration')]
    panels=[(axes[1,0],'root_roll_deg','TRAIN69: roll','deg'),
            (axes[1,1],'root_pitch_deg','TRAIN69: pitch','deg'),
            (axes[2,0],'root_angular_speed_rad_s','TRAIN69: angular speed','rad/s'),
            (axes[2,1],'supported_foot_count','TRAIN69: observed supported feet','count')]
    for ax,field,title,unit in panels:
        for key,color,label in comparisons:
            t=public[key+'_time_relative_landing_s'];v=public[key+'_'+field];keep=(t>=-.2)&(t<=2.6)
            if field=='supported_foot_count':ax.step(t[keep],v[keep],where='post',color=color,lw=1.1,alpha=.8,label=label)
            else:ax.plot(t[keep],v[keep],color=color,lw=1.5,label=label)
        ax.set_title(title);ax.set_ylabel(unit);ax.set_xlim(-.2,2.6)
        if field in ['root_roll_deg','root_pitch_deg']:
            ax.axhline(15.,color='gray',lw=.8,ls=':');ax.axhline(-15.,color='gray',lw=.8,ls=':')
            # These bounds apply during the later continuous stable window,
            # not at the instant of first foot landing.
        if field=='root_angular_speed_rad_s':ax.axhline(1.,color='gray',lw=.8,ls=':')
        if field=='supported_foot_count':ax.set_ylim(-.15,4.2);ax.set_yticks(range(5))
    axes[1,0].legend(fontsize=8,loc='lower left')
    for ax in axes.flat:
        ax.axvline(0.,color='black',lw=.9,ls='--',alpha=.6)
        ax.grid(alpha=.16);ax.set_xlabel('Seconds relative to each native contact-flight landing')
    fig.suptitle('Actual contact-flight landing and recovery signals\n'
        'Recovered reference can land tilted; sustained support and settling differ. No new native trial.',fontsize=14)
    out=HERE/'LANDING_AND_SUPPORT.png';assert not out.exists()
    fig.savefig(out,dpi=135,facecolor='white');plt.close(fig)
    files=[Path(__file__),HERE/'ACTUAL_POSTURE_CHECK.json',HERE/'PUBLIC_STABILITY_CHECK.json',*paths]
    record=dict(rendered_at=datetime.now(timezone.utc).isoformat(),figure_sha256=sha(out),
        source_sha256={str(p):sha(p) for p in files},actual_pose_and_public_data_only=True,
        original_rules_and_native_outcomes_unchanged=True,new_native_execution=False,formal_acceptance=False)
    with (HERE/'FIGURE_RECEIPT.json').open('x') as f:f.write(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(figure=str(out),bytes=out.stat().st_size,sha256=sha(out))))


if __name__=='__main__':main()
