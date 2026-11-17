"""Conservative necessary reach bound for an authored serial arm/string strip.

Triangle inequality ignores joint limits, collisions and dynamics; a point
inside the sphere is NOT certified reachable. An outside point cannot be
reached at this base pose/time within the specified strip by any joint angles.
"""
import itertools
import numpy as np


def string_strip_reach_bound(geometry,target_body,strip,*,ball_radius_m,face_sign=1.,position_budget_m=.0005):
    target=np.asarray(target_body,dtype=float)
    center=np.asarray(strip['center'],dtype=float);half=.5*np.asarray(strip['size'],dtype=float)
    if (target.shape!=(3,) or center.shape!=(3,) or half.shape!=(3,) or
        not np.isfinite(np.r_[target,center,half,ball_radius_m,position_budget_m]).all() or
        np.any(half<=0) or ball_radius_m<=0 or position_budget_m<0 or face_sign not in (-1.,1.)):
        raise ValueError('Finite authored strip and contact geometry required')
    shoulder=np.asarray(geometry.arm_origin)+np.asarray(geometry.joints[0][0])
    link_bound=sum(np.linalg.norm(offset) for offset,_ in geometry.joints[1:])
    # The norm of an affine point over a rectangle is convex: its maximum
    # occurs at a vertex. Include the original ball-center surface offset.
    proxy_x=center[0]+face_sign*(half[0]+ball_radius_m)
    end_lengths=[]
    for sy,sz in itertools.product((-1.,1.),repeat=2):
        proxy=np.array([proxy_x,center[1]+sy*half[1],center[2]+sz*half[2]])
        end_lengths.append(np.linalg.norm(geometry.tool_offset+geometry.tool_rotation@proxy))
    upper=float(link_bound+max(end_lengths))
    distance=float(np.linalg.norm(target-shoulder));deficit=max(0.,distance-upper-position_budget_m)
    return dict(shoulder_body_m=shoulder.tolist(),target_distance_m=distance,
        optimistic_reach_upper_bound_m=upper,position_budget_m=position_budget_m,
        outside_reach_bound=bool(deficit>0),conditional_translation_lower_bound_m=deficit,
        bound_scope='fixed base pose/time; current authored string strip; all joint angles',
        feasible_if_inside=False,dynamics_verified=False,
        unrealized_base_translation_only=True)
