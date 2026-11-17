"""Analytic task margins and constrained goals; never a native certificate."""
import numpy as np
from scipy.optimize import minimize
from quadsmash.physics.court_v4 import top_height

FLAGS=('contact_feasible','legal_return_predicted','attack5_predicted','target_landing_predicted')
SCALES=np.array([5.,1.,5.,.1,1.,1.])
NAMES=('speed','downward','forward','net','court','target')


def task_margins(outcome,court):
    """Physical margins of the FORWARD collision; missing evidence fails closed."""
    v=np.asarray(outcome.get('outgoing_velocity'),dtype=float)
    land=np.asarray(outcome.get('landing_position'),dtype=float)
    margins=np.full(6,-1e3)
    if v.shape==(3,) and np.isfinite(v).all():
        margins[:3]=[np.linalg.norm(v)-5.,-1.-v[2],v[0]]
    net=outcome.get('net_clearance')
    cross=np.asarray(outcome.get('crossing_position'),dtype=float)
    if net is not None and np.isfinite(net) and cross.shape==(3,) and np.isfinite(cross).all():
        margins[3]=min(float(net),court['net_width_m']/2-abs(cross[1]))
    if land.shape==(3,) and np.isfinite(land).all():
        x,y=land[:2];lo,hi=court['opponent_x_limits_m']
        margins[4]=min(x-lo,hi-x,court['singles_width_m']/2-abs(y))
        lo,hi=court['target_x_limits_m']
        margins[5]=min(x-lo,hi-x,court['target_y_half_width_m']-abs(y))
    normalized=margins/SCALES
    return dict(physical=dict(zip(NAMES,margins.tolist())),scales=dict(zip(NAMES,SCALES.tolist())),
        normalized=normalized,gamma=float(normalized.min()),limiting_constraint=NAMES[int(normalized.argmin())],
        valid=bool(np.all(normalized>=0) and all(outcome.get(k) is True for k in FLAGS)))


def attack_geometry(position,physics,court,outcome=None):
    p=np.asarray(position,dtype=float)
    if p.shape!=(3,) or not np.isfinite(p).all():raise ValueError('Finite world contact position required')
    radius=physics['collision_radius_m'];minimum=court['net_center_top_m']+radius
    # Necessary only, under the configured gravity/drag with no upward wind.
    wind=physics.get('wind_m_s',physics.get('wind_velocity_mps',[0.,0.,0.]))
    wind=np.asarray(wind,dtype=float)
    applicable=bool(wind.shape==(3,) and wind[2]<=0)
    own=bool(p[0]<court['net_plane_x_m'])
    possible=bool(own and (not applicable or p[2]>minimum))
    crossing=None if outcome is None else outcome.get('crossing_position')
    height=None;drop=None
    if crossing is not None:
        crossing=np.asarray(crossing);height=float(top_height(crossing[1],court)+radius);drop=float(p[2]-crossing[2])
    return dict(predicted_contact_height=float(p[2]),distance_to_net_m=float(court['net_plane_x_m']+radius-p[0]),
        required_net_crossing_height=minimum if height is None else height,
        predicted_vertical_drop_to_net=drop,downward_attack_geometrically_possible=possible,
        necessary_condition_only=True,monotone_height_assumption_applicable=applicable,
        rejection=None if possible else 'DOWNWARD_ATTACK_HEIGHT_INFEASIBLE' if own else 'CONTACT_NOT_ON_OWN_SIDE',
        unrealized_takeoff_requirements=dict(contact_height_increase_lower_bound_m=max(0.,minimum-p[2]),
            contact_time_change_s=None,base_linear_velocity_change_m_s=None,pitch_change_rad=None,
            angular_velocity_change_rad_s=None,implemented=False,
            reason='Height is a necessary lower bound; other minimum changes cannot be inferred from a local failed solve.'))


def optimize_task_goal(model,position,incoming,local,max_iterations=24):
    """Max-min task margins, constrained on actual forward outcome, one seed."""
    from .inverse_impact_target_v1 import inverse_impact_target
    p=np.asarray(position);incoming=np.asarray(incoming);geometry=attack_geometry(p,model.physics,model.court)
    seed=inverse_impact_target(model,p,incoming,local) if geometry['downward_attack_geometrically_possible'] else dict(
        desired_outgoing_velocity_world_m_s=[5.,0.,-1.],normal_world=[1.,0.,0.],racket_velocity_world_m_s=[0.,0.,0.],accepted=False)
    cache={};best=None;best_gamma=-np.inf
    def evaluate(x):
        nonlocal best,best_gamma
        key=np.asarray(x[:3]).tobytes()
        if key not in cache:
            delta=x[:3]-incoming;n=delta/max(np.linalg.norm(delta),1e-12)
            vr=incoming+delta/(1+model.physics['restitution'])
            out=model.evaluate(p,incoming,p-model.physics['collision_radius_m']*n,vr,n,local)
            m=task_margins(out,model.court);cache[key]=(n,vr,out,m)
            if m['valid'] and m['gamma']>best_gamma:best=(x[:3].copy(),n.copy(),vr.copy(),out,m);best_gamma=m['gamma']
        return cache[key]
    start=np.r_[seed['desired_outgoing_velocity_world_m_s'],0.];start[:3]=np.clip(start[:3],[.01,-30,-20],[40,30,-1])
    first=evaluate(start);start[3]=max(-1000.,first[3]['gamma'])
    fit=None
    if geometry['downward_attack_geometrically_possible']:
        fit=minimize(lambda x:-x[3],start,method='SLSQP',bounds=[(.01,40),(-30,30),(-20,-1),(-1000,2)],
            constraints=[dict(type='ineq',fun=lambda x:evaluate(x)[3]['normalized']-x[3])],
            options=dict(maxiter=max_iterations,ftol=1e-7,eps=1e-4))
        evaluate(fit.x)
    candidate=start if fit is None else fit.x;n,vr,out,m=evaluate(candidate)
    if best is not None:candidate,n,vr,out,m=best
    return dict(accepted=best is not None,analytic_target_valid=best is not None,
        accepted_semantics='forward_full_ball_task_only',initialization_only=best is None,
        normal_world=n.tolist(),racket_velocity_world_m_s=vr.tolist(),
        desired_outgoing_velocity_world_m_s=np.asarray(candidate[:3]).tolist(),contact_point_local=np.asarray(local).tolist(),
        forward_outcome=out,task_margins=m,attack_geometry=attack_geometry(p,model.physics,model.court,out),original_target=seed,
        solver_success=False if fit is None else bool(fit.success),queries=len(cache),
        robot_contact_kinematic_valid=False,robot_feasibility_verified=False,
        rejection_reason=None if best is not None else m['limiting_constraint'])
