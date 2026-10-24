"""TRAIN guidance from actual release/first landing; native task flags stay separate."""
import numpy as np
from .full_racket_contact_curriculum_v4 import usable_air_contact


def norm_outgoing_shaping(row):
    if not usable_air_contact(row) or not row['released']:
        return 0.
    velocity=np.asarray(row['outgoing_velocity_mps'],dtype=float)
    if velocity.shape!=(3,) or not np.isfinite(velocity).all() or velocity[0]<=0:
        return 0.
    # The actual task requires norm>=8, not forward component>=8. This
    # permits a steeper, shorter attack with e.g.vx7.55andvz-2.35.
    return float(10*np.clip(np.linalg.norm(velocity)/8.,0.,1.)+3*np.clip(-velocity[2],0.,1.))


def target_landing_shaping(row,target_x,target_y):
    target_x=np.asarray(target_x,dtype=float);target_y=float(target_y)
    if (target_x.shape!=(2,) or not np.isfinite(target_x).all() or target_x[0]>=target_x[1]
            or not np.isfinite(target_y) or target_y<=0):
        raise ValueError('Require the finite ordered original target bounds')
    if not (usable_air_contact(row) and row['released'] and row['crossed_net'] and row['ground_seen']):
        return 0.
    point=np.asarray(row['first_landing_m'],dtype=float)
    if point.shape!=(3,) or not np.isfinite(point).all():
        return 0.
    distance=np.hypot(max(target_x[0]-point[0],0.,point[0]-target_x[1]),
                      max(abs(point[1])-target_y,0.))
    interior=min(point[0]-target_x[0],target_x[1]-point[0],target_y-abs(point[1]))
    # 35cm distance scale and15cm interior preference are TRAIN shaping,
    # never replacement landing bounds or acceptance tolerances.
    return float(20*np.exp(-(distance/.35)**2)+5*np.clip(interior/.15,0.,1.))
