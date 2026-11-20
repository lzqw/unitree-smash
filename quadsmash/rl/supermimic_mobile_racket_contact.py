"""Instantaneous response of a free articulated racket to simultaneous shuttle contact impulses."""
import torch

from .supermimic_articulated_inertia_v1 import skew


def local_racket_mobility(model, joints, quaternion):
    """Recover the spatial Jacobian at the racket origin from point Jacobians."""
    zero = torch.zeros_like(joints[:,:3])
    _, at_origin = model(joints,quaternion,zero)
    geometry = model.geometry(joints,torch.zeros_like(joints),zero,quaternion,zero,zero)
    rotation = geometry['rotation'][:,model.geometry.ids['Racket']]
    origin_jac = at_origin['point_jacobian']
    angular_jac = torch.zeros_like(origin_jac)
    for axis in torch.eye(3,device=joints.device,dtype=joints.dtype):
        _, at_point = model(joints,quaternion,axis.expand_as(zero))
        displacement = (rotation @ axis)
        angular_jac += .5*skew(displacement) @ (at_point['point_jacobian']-origin_jac)
    Rt = rotation.transpose(-1,-2)
    jac = torch.cat((Rt @ origin_jac,Rt @ angular_jac),1)
    return jac @ torch.linalg.solve(at_origin['mass_matrix'],jac.transpose(-1,-2))


def solve_mobile_patch(normals,mu,restitution,velocity,spin,surface,racket_arm,radius,active,
                       normal_scale,iterations,spatial,shuttle_mass,separation_velocity=None,load_trace=None):
    # Work in velocity units (impulse divided by shuttle mass). One shared
    # twist carries reaction from one loaded racket shape to the other shapes.
    n, shapes, _ = normals.shape
    eye=torch.eye(3,device=velocity.device,dtype=velocity.dtype).expand(n,shapes,-1,-1)
    wrenches=torch.cat((eye,skew(racket_arm)),dim=-2)
    twist_per_load=shuttle_mass * (spatial[:,None] @ wrenches)
    point_inverse=wrenches.transpose(-1,-2) @ twist_per_load
    lever=-radius*normals
    incoming=velocity[:,None]+torch.linalg.cross(spin[:,None].expand_as(lever),lever)-surface
    closing=(incoming*normals).sum(-1)
    desired=closing+(1+restitution)*normal_scale*(-closing).clamp_min(0)
    if separation_velocity is not None:
        desired=torch.maximum(closing,desired-separation_velocity)
    normal_inverse=1+(normals[...,None,:] @ point_inverse @ normals[...,None]).squeeze(-1).squeeze(-1)
    plane=eye-normals[...,None]*normals[...,None,:]
    tangent_inverse=torch.linalg.inv(3.5*eye+plane @ point_inverse @ plane)
    loads=list(torch.zeros_like(mu).unbind(1))
    tangents=list(torch.zeros_like(normals).unbind(1))
    linear,angular=velocity,spin
    reaction=torch.cat((torch.zeros_like(velocity),torch.zeros_like(spin)),-1)
    for _ in range(iterations):
        for j in range(shapes):
            normal=normals[:,j]
            relative=(linear+torch.linalg.cross(angular,lever[:,j])-surface[:,j]
                      -(wrenches[:,j].transpose(-1,-2) @ reaction[...,None]).squeeze(-1))
            vn=(relative*normal).sum(-1)
            load=torch.where(active[:,j],(loads[j]+(desired[:,j]-vn)/normal_inverse[:,j]).clamp_min(0),0.)
            normal_change=(load-loads[j])[:,None]*normal
            loads[j]=load
            # The racket's anisotropic normal reaction also changes tangential slip.
            post_normal=relative+normal_change+(point_inverse[:,j] @ normal_change[...,None]).squeeze(-1)
            slip=post_normal-(post_normal*normal).sum(-1,keepdim=True)*normal
            free=tangents[j]-(tangent_inverse[:,j] @ slip[...,None]).squeeze(-1)
            bounded=free*((load*mu[:,j])/free.norm(dim=-1).clamp_min(1.e-12)).clamp(max=1)[:,None]
            change=bounded-tangents[j]
            tangents[j]=bounded
            total=normal_change+change
            linear=linear+total
            angular=angular+torch.linalg.cross(lever[:,j],change)/(.4*radius**2)
            reaction=reaction-(twist_per_load[:,j] @ total[...,None]).squeeze(-1)
    normal_loads=torch.stack(loads,1)
    friction=torch.stack(tangents,1)
    if load_trace is not None:
        sliding=friction.norm(dim=-1)>=(mu*normal_loads-1.e-8).clamp_min(0)
        load_trace['regime']=(normal_loads>1.e-8).to(torch.uint8)*(1+sliding.to(torch.uint8))
    return (normal_loads[...,None]*normals).sum(1),friction.sum(1),angular
