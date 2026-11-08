"""Experimental finite-iteration impact probe, not a PhysX replacement.

Inspired by pinned PhysX 5.4 TGS source: accumulated displacement bias,
paired tangential impulses, position iterations followed by velocity passes.
The inertia/Jacobians and bounded motor estimate stay frozen within one
physical interval. Persistent friction anchors and articulation internals
are absent. Only a diagnostic until measured comparisons justify more work.
"""
import copy
import numpy as np
import pinocchio as pin

from scripts.audit_sphere_touchdown_predictions_v1 import sphere_gaps
from scripts.predict_speed_limited_spheres_v1 import surface_jacobians
from scripts.frictional_speed_limited_step_v1 import speed_limited_velocity_step


def temporal_impact_probe(model,interface,state,target,spheres,dt):
    if not 0<dt<=.0001:raise ValueError('Physical-step diagnostic only')
    target=np.asarray(target,float)
    if target.shape!=(18,) or not np.isfinite(target).all():raise ValueError('Invalid target')
    snapshot=model.snapshot(state.q,state.v);mass=snapshot['mass_matrix'];bias=snapshot['nonlinear']
    gap=sphere_gaps(model,state.q,spheres);world_jac=surface_jacobians(model,state.q,spheres)
    selection=np.zeros((24,18));selection[model.v_indices,np.arange(18)]=1.
    original=speed_limited_velocity_step(mass,state.v,bias,selection,interface.kp,interface.kd,
        target-state.q[model.q_indices],interface.effort,dt,world_jac.reshape(12,24),
        -gap/np.where(gap>0,dt,.01),interface.friction,interface.velocity)
    def following(q,v,impulse):
        result=copy.deepcopy(state);result.q=q;result.v=v
        result.root_position_world=q[:3].copy();result.root_rotation_world=pin.Quaternion(q[3:7]).matrix()
        result.joint_position=q[model.q_indices].copy();result.joint_velocity=v[model.v_indices].copy()
        result.contacts=impulse[:,2]>1e-8
        return result
    impulse=original['impulse'].reshape(4,3)
    unchanged=dict(state=following(pin.integrate(model.model,state.q,dt*original['velocity']),
        original['velocity'],impulse),impulse=impulse,original_step_exact=True,
        momentum_residual=original['momentum_residual'])
    if not np.any((gap>1e-6)&(impulse[:,2]>1e-10)):return unchanged
    # Candidate basis uses contact-point incoming velocity. Native source uses
    # link linear velocity for its basis; that unimplemented distinction is
    # deliberately not presented as exact TGS equivalence.
    rotations=[]
    for j in world_jac:
        tangent=(j@state.v)[:2];norm=np.linalg.norm(tangent)
        tangent=tangent/norm if norm>.01 else np.array([0.,-1.])
        rotations.append(np.array([[*tangent,0.],[-tangent[1],tangent[0],0.],[0.,0.,1.]]))
    rotations=np.asarray(rotations)
    jac=np.einsum('fij,fjk->fik',rotations,world_jac).reshape(12,24)
    response=np.linalg.solve(mass,jac.T);matrix=jac@response
    motor=selection@original['torque']
    velocity=state.v+dt*np.linalg.solve(mass,motor-bias)
    displacement=np.zeros(24);applied=np.zeros(12);subdt=dt/8.
    # Explicit candidate numerical settings, not changes to native physics.
    friction_scale=.8;normal_bias=.8
    for iteration in range(10):
        motion=(jac@displacement).reshape(4,3)
        for foot in range(4):
            k=3*foot;normal=k+2;tangent=slice(k,k+2)
            sep=gap[foot]+motion[foot,2]
            if iteration>=8:sep=max(0.,sep)
            coefficient=1./(dt+subdt) if gap[foot]>0 else normal_bias/subdt
            desired=-coefficient*sep
            new=max(0.,applied[normal]+(desired-jac[normal]@velocity)/matrix[normal,normal])
            velocity+=response[:,normal]*(new-applied[normal]);applied[normal]=new
            tangent_bias=friction_scale*motion[foot,:2]/subdt
            proposed=applied[tangent]-(jac[tangent]@velocity+tangent_bias)/np.diag(matrix)[tangent]
            limit=interface.friction*new;length=np.linalg.norm(proposed)
            if length>limit:proposed*=limit/max(length,1e-30)
            velocity+=response[:,tangent]@(proposed-applied[tangent]);applied[tangent]=proposed
        if iteration<8:displacement+=subdt*velocity
    world_impulse=np.einsum('fji,fj->fi',rotations,applied.reshape(4,3))
    residual=mass@(velocity-state.v)+dt*bias-dt*motor-jac.T@applied
    q=pin.integrate(model.model,state.q,displacement)
    return dict(state=following(q,velocity,world_impulse),impulse=world_impulse,
        original_step_exact=False,momentum_residual=float(np.max(abs(residual))),
        joint_speed_excess=float(np.max(abs(velocity[model.v_indices])-interface.velocity)),
        position_iterations=8,velocity_iterations=2)
