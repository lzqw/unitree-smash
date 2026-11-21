"""Test whether measured separation in units of one incident substep explains friction gain."""
import argparse
import collections
import json
import math
import statistics
from pathlib import Path

from evaluate_supermimic_contact_rollout import sha


def rows(history, dt):
    panel=[]
    for c in history['cases']:
        if c['mode']!='racket_stable':
            continue
        signs={}
        for name,events in c['branches'].items():
            if len(events)!=1:
                break
            e=events[0]
            n=e['normal_dv_mps']
            load=math.hypot(*n)
            speeds=e['tangent_speed_pre_post_mps'][0]
            if not speeds or not e['separation_m'] or load==0:
                break
            closing=-statistics.mean(sum(x*y for x,y in zip(v,n,strict=True))/load for v in e['relative_velocity_w_mps'][0])
            if closing<=0:
                break
            gap=statistics.mean(e['separation_m'])
            signs[name]=dict(gap_m=gap,closing_mps=closing,gap_fraction=max(-1,min(1,gap/(dt*closing))),
                             speed=statistics.mean(speeds),normal_load=load,observed=math.hypot(*e['tangent_dv_mps']))
        if set(signs)=={'minus','plus','nominal_a','nominal_b'}:
            panel.append(dict(lane=c['lane'],branches=signs))
    return panel


def evaluate(panel, slope, friction, sticking):
    scored=[]
    for case in panel:
        values=case['branches']
        def predict(b):
            v=values[b]
            return min(friction*v['normal_load'],sticking*v['speed']*(1+slope*v['gap_fraction']))
        lo,hi=(values[b]['observed'] for b in ('minus','plus'))
        actual=hi-lo
        pred=predict('plus')-predict('minus')
        scored.append(dict(lane=case['lane'],observed=actual,predicted=pred,
                           above_noise=abs(actual)>max(1e-8,3*abs(values['nominal_a']['observed']-values['nominal_b']['observed'])),
                           branch_mse=((predict('plus')-hi)**2+(predict('minus')-lo)**2)/2))
    return dict(signed_rmse=math.sqrt(statistics.mean((r['observed']-r['predicted'])**2 for r in scored)),
                absolute_rmse=math.sqrt(statistics.mean(r['branch_mse'] for r in scored)),
                sign_agreements=sum(r['above_noise'] and r['observed']*r['predicted']>0 for r in scored),
                above_noise=sum(r['above_noise'] for r in scored),cases=scored)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('training','development'):
        parser.add_argument('--'+name,type=Path,nargs='+',required=True)
    for name in ('fit','physics','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args()
    read=lambda p:json.loads(p.read_text())
    fit=read(a.fit)
    if sha(a.physics)!=fit['physics_sha256']:
        raise ValueError('Shuttle configuration changed')
    # The fit binds the simulator data and physics dt through the G07 source.
    dt=read(a.physics)['native_physics_dt_s']
    groups={}
    for role,paths in [('training',a.training),('development',a.development)]:
        for path in paths:
            history=read(path)
            key=history['source_sha256']
            if history['role']!=role or key in groups:
                raise ValueError('Wrong role or repeated native prefix')
            records=rows(history,dt)
            if not records:
                raise ValueError('Missing measured force records')
            groups[key]=dict(role=role,file_sha256=sha(path),modes=dict(collections.Counter(r['mode'] for r in history['cases'])),
                             data=records)
    cal=fit['selected']
    def score(k,s):
        return evaluate(groups[k]['data'],s,cal['friction'],cal['sticking'])
    train=[k for k,g in groups.items() if g['role']=='training']
    base={k:score(k,0) for k in groups}
    def choose(panels):
        def loss(s):
            scores=[(score(k,s),base[k]) for k in panels]
            return statistics.mean((r[m]/max(1e-8,b[m]))**2 for r,b in scores for m in ('signed_rmse','absolute_rmse'))
        return min((i/8 for i in range(-8,9)),key=lambda s:(loss(s),abs(s)))
    if len(train)<2:
        raise ValueError('Need independent training prefixes')
    fitted=choose(train)
    for k,g in groups.items():
        g['baseline']=base[k]
        g['fitted']=score(k,fitted)
        if k in train:
            slope=choose([j for j in train if j!=k])
            g['leave_prefix_out']=dict(slope=slope,metrics=score(k,slope))
    result=dict(fit_sha256=sha(a.fit),slope=fitted,groups=groups,scope='Measured single-load friction magnitude only',
                actor_updates=0,v13_updates=0,admitted=False)
    with a.output.open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False)


if __name__=='__main__':
    main()
