"""Rank full physical recovery before sampled surface-margin improvements.

Uses the unchanged fine dynamics/limits and original final stability window.
Margins are measured at 10 ms boundaries, never presented as a substep or
native certificate. Full18 commands remain adjustable from the actual release.
"""
import time
import numpy as np
from scripts.dense_reference_correction_rollout_v1 import rollout as dense_rollout, initialize, knots_for, WORK


def rollout(key,nodes,deadline):
    started=time.monotonic()
    result=dense_rollout(key,nodes,160,deadline)
    controller=WORK['c'];record=WORK['records'][key]
    nonfoot=[i for i,s in enumerate(controller.geometry.shapes)
             if not s['body'].endswith('_foot') and '::' not in s['body']]
    head=[i for i,s in enumerate(controller.geometry.shapes) if s['body'] in ['Head_lower','Head_upper']]
    if not nonfoot or not head:raise ValueError('Original nonfoot and head geometry required')
    minimum=float('inf');head_minimum=float('inf');penalty=0.;at=None
    for step,q in [(-1,record['q'])]+[(r['step'],r['q']) for r in result['trajectory']]:
        gaps,_=controller.geometry.evaluate(np.asarray(q))
        local=int(nonfoot[int(np.argmin(gaps[nonfoot]))])
        if gaps[local]<minimum:
            minimum=float(gaps[local]);at=dict(step=step,body=controller.geometry.shapes[local]['body'])
        head_minimum=min(head_minimum,float(np.min(gaps[head])))
        penalty+=.01*float(np.sum(np.maximum(0.,(.01-gaps[nonfoot])/.01)**2))
    full=result['model_path_complete'];stable=full and result['terminal_stable_s']>=.8
    result.update(sampled_minimum_nonfoot_m=minimum,sampled_minimum_head_m=head_minimum,
        sampled_minimum_location=at,sampled_margin_penalty=penalty,complete_stable_model=stable,
        rank=[0 if stable else (1 if full else 2),result['required_steps']-result['steps'],
              max(0.,.8-result['terminal_stable_s']),max(0.,.01-minimum),penalty,result['cost']],
        worker_wall_seconds=time.monotonic()-started,surface_sample_period_s=.01)
    return result
