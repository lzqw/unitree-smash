"""Traceable full-racket geometry and compound inertia, independent of Isaac."""
from pathlib import Path
import json
import math
import os
from functools import lru_cache

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _config_root():
    configured = os.environ.get("QUADSMASH_G07_EXECUTION_ROOT")
    return Path(configured).expanduser().resolve() if configured else ROOT


def load_racket(config_path='configs/full_racket_v4_r3.json'):
    return json.loads((_config_root()/config_path).read_text())


def apply_neutral_arm_reset(cfg):
    reset=json.loads((_config_root()/'configs/full_racket_reset_v4.json').read_text())
    cfg.robot.init_state.joint_pos.update({f'joint{i+1}':q for i,q in enumerate(reset['arm_joint_positions_rad'])})
    return cfg


def string_polygon(config):
    a, b = config['assumptions']['strings_semiaxes_yz_m']
    angles = np.arange(config['strings_polygon_sides'])*2*math.pi/config['strings_polygon_sides']
    return np.stack((a*np.cos(angles), b*np.sin(angles)), -1)


def string_prism_mesh(config):
    polygon = string_polygon(config)
    n = len(polygon)
    h = config['assumptions']['string_thickness_m']/2
    points = [(-h, float(y), float(z)) for y, z in polygon]
    points += [(h, float(y), float(z)) for y, z in polygon]
    faces = [tuple(reversed(range(n))), tuple(range(n, 2*n))]
    faces += [(i, (i+1)%n, (i+1)%n+n, i+n) for i in range(n)]
    return points, faces


def string_strips(config):
    """Inscribed boxes; no strip extends beyond the frozen32gon string area."""
    return _string_strips(tuple(config['assumptions']['strings_semiaxes_yz_m']),
                         config['strings_polygon_sides'],config['assumptions']['string_thickness_m'],
                         config['strings_strip_count'])


@lru_cache(maxsize=16)
def _string_strips(semiaxes,sides,thickness,count):
    config=dict(assumptions=dict(strings_semiaxes_yz_m=semiaxes),strings_polygon_sides=sides)
    polygon=string_polygon(config)
    edges=np.roll(polygon,-1,axis=0)-polygon
    zmax=semiaxes[1]
    boundaries=np.linspace(-zmax,zmax,count+1)
    result=[]
    def half_width(z):
        # The polygon is CCW. All edges constrain the permitted positive y.
        bounds=[]
        for point,edge in zip(polygon,edges):
            if edge[1]>1e-14:bounds.append(point[0]+edge[0]*(z-point[1])/edge[1])
        return max(0.,min(bounds))
    for lower,upper in zip(boundaries[:-1],boundaries[1:]):
        width=min(half_width(lower),half_width(upper))
        if width<=1e-12:continue
        result.append(dict(center=[0.,0.,(lower+upper)/2],
            size=[thickness,2*width,upper-lower]))
    return result


@lru_cache(maxsize=32)
def _strip_tensors(semiaxes,sides,thickness,count,device,dtype):
    import torch
    strips=_string_strips(semiaxes,sides,thickness,count)
    # Avoid rebuilding and copying the same62 boxes to the GPU at each
    # native substep. Cache only geometry constants, never measured states.
    with torch.inference_mode(False):
        centers=torch.tensor([s['center'] for s in strips],device=device,dtype=dtype)
        half=torch.tensor([s['size'] for s in strips],device=device,dtype=dtype)/2
    return centers,half


def cylinder_inertia(mass, radius, length, axis, inner_radius=0.):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    squared_radius = radius*radius+inner_radius*inner_radius
    axial = mass*squared_radius/2
    perpendicular = mass*(3*squared_radius+length*length)/12
    return perpendicular*np.eye(3)+(axial-perpendicular)*np.outer(axis, axis)


def parts(config):
    a, p = config['assumptions'], config['published']
    head = config['head_center_from_butt_m']
    handle_length = p['handle_length_m']
    frame_half = a['head_outer_length_m']/2
    shaft_length = head-frame_half-handle_length
    shaft_center = handle_length+shaft_length/2
    # Published balance constrains the unstrung representative distribution.
    remaining = a['unstrung_mass_kg']-a['shaft_mass_kg']
    frame_mass = (a['unstrung_mass_kg']*p['unstrung_balance_from_butt_m']
                  -a['shaft_mass_kg']*shaft_center-remaining*handle_length/2)/(head-handle_length/2)
    handle_mass = remaining-frame_mass
    result = []
    def add(name, kind, mass, center, inertia, **shape):
        if mass <= 0:
            raise ValueError('Every physical component must have positive mass')
        result.append(dict(name=name, kind=kind, mass=mass, center=np.asarray(center),
                           inertia=np.asarray(inertia), **shape))
    add('Handle', 'cylinder', handle_mass, [0, 0, handle_length/2-head],
        cylinder_inertia(handle_mass, a['handle_radius_m'], handle_length, [0,0,1]),
        radius=a['handle_radius_m'], length=handle_length, axis=np.array([0.,0.,1.]), collision=False)
    add('Grip', 'cylinder', a['added_grip_mass_kg'], [0,0,handle_length/2-head],
        cylinder_inertia(a['added_grip_mass_kg'], a['handle_radius_m']+.0005, handle_length,
                         [0,0,1], a['handle_radius_m']),
        radius=a['handle_radius_m']+.0005, length=handle_length, axis=np.array([0.,0.,1.]), collision=True)
    add('Shaft', 'cylinder', a['shaft_mass_kg'], [0,0,shaft_center-head],
        cylinder_inertia(a['shaft_mass_kg'], a['shaft_diameter_m']/2, shaft_length, [0,0,1]),
        radius=a['shaft_diameter_m']/2, length=shaft_length, axis=np.array([0.,0.,1.]), collision=True)
    angles = np.arange(config['frame_segments']+1)*2*math.pi/config['frame_segments']
    radius = a['frame_tube_radius_m']
    ring = np.stack((np.zeros_like(angles), (a['head_outer_width_m']/2-radius)*np.cos(angles),
                     (frame_half-radius)*np.sin(angles)), -1)
    lengths = np.linalg.norm(np.diff(ring, axis=0), axis=-1)
    for index, length in enumerate(lengths):
        mass = frame_mass*length/lengths.sum()
        axis = (ring[index+1]-ring[index])/length
        add(f'Frame_{index:02d}', 'capsule', mass, (ring[index]+ring[index+1])/2,
            cylinder_inertia(mass, radius, float(length), axis), radius=radius,
            length=float(length), axis=axis, collision=True)
    polygon = string_polygon(config)
    y, z = polygon.T
    yn, zn = np.roll(y,-1), np.roll(z,-1)
    cross = y*zn-yn*z
    area = cross.sum()/2
    yy = (cross*(y*y+y*yn+yn*yn)).sum()/(12*area)
    zz = (cross*(z*z+z*zn+zn*zn)).sum()/(12*area)
    yz = (cross*(2*y*z+y*zn+yn*z+2*yn*zn)).sum()/(24*area)
    second = np.array([[a['string_thickness_m']**2/12,0,0],[0,yy,yz],[0,yz,zz]])
    if config.get('string_collision_representation')=='inscribed-z-strips':
        strips=string_strips(config)
        volumes=np.array([np.prod(s['size']) for s in strips])
        second=sum(volume*(np.diag(np.array(s['size'])**2/12)+np.outer(s['center'],s['center']))
                   for volume,s in zip(volumes,strips))/volumes.sum()
    add('Collider', 'string_prism', a['strings_mass_kg'], [0,0,0],
        a['strings_mass_kg']*(np.trace(second)*np.eye(3)-second), collision=True)
    size = np.array(a.get('clamp_size_m',[.045,.055,.060]))
    mass = a['clamp_mass_kg']
    add('Clamp', 'box', mass, a.get('clamp_center_in_body_m',[-.0225,0,config['grip_center_from_butt_m']-head]),
        np.diag(mass*(size.dot(size)-size*size)/12), size=size, collision=True)
    return result


def mass_properties(config):
    components = parts(config)
    mass = sum(part['mass'] for part in components)
    com = sum(part['mass']*part['center'] for part in components)/mass
    inertia = np.zeros((3,3))
    for part in components:
        offset = part['center']-com
        inertia += part['inertia']+part['mass']*(offset.dot(offset)*np.eye(3)-np.outer(offset,offset))
    values, axes = np.linalg.eigh(inertia)
    if np.linalg.det(axes) < 0:
        axes[:,0] *= -1
    return dict(mass_kg=mass, center_of_mass_in_body_m=com.tolist(), inertia_in_body_kg_m2=inertia.tolist(),
                principal_inertia_kg_m2=values.tolist(), principal_axes_in_body=axes.tolist(),
                component_count=len(components))


class RacketPlaneSupport:
    """Exact +X support of the declared USD racket primitives, without clamp.

    Unlike rotated local AABB corners, these extents cannot invent racket
    material beyond the court plane. No dynamics, state cache or forces.
    """
    def __init__(self,config,device='cpu',dtype=None):
        import torch
        dtype=torch.float32 if dtype is None else dtype
        centers,axes,halflengths,radii,kinds,box_halves=[],[],[],[],[],[]
        def add(center,axis,length,radius,kind,half):
            centers.append(center);axes.append(axis);halflengths.append(length/2)
            radii.append(radius);kinds.append(kind);box_halves.append(half)
        for part in parts(config):
            if part['name']=='Clamp' or not part['collision']:continue
            if part['kind']=='string_prism':
                if config.get('string_collision_representation')!='inscribed-z-strips':
                    raise ValueError('Plane support requires the native strip racket version')
                for strip in string_strips(config):
                    add(strip['center'],[0,0,1],0,0,2,np.asarray(strip['size'])/2)
            else:
                if part['kind'] not in ('capsule','cylinder'):
                    raise ValueError('Unsupported native racket primitive')
                add(part['center'],part['axis'],part['length'],part['radius'],
                    0 if part['kind']=='capsule' else 1,[0,0,0])
        with torch.inference_mode(False):
            self.centers=torch.tensor(np.asarray(centers),device=device,dtype=dtype)
            self.axes=torch.tensor(np.asarray(axes),device=device,dtype=dtype)
            self.halflengths=torch.tensor(halflengths,device=device,dtype=dtype)
            self.radii=torch.tensor(radii,device=device,dtype=dtype)
            self.kinds=torch.tensor(kinds,device=device)
            self.box_halves=torch.tensor(np.asarray(box_halves),device=device,dtype=dtype)

    def maximum_x(self,position,quaternion):
        import torch
        from .native import quat_rotate
        inverse=torch.cat((quaternion[...,:1],-quaternion[...,1:]),-1)
        direction=torch.zeros_like(position);direction[...,0]=1.
        local_x=quat_rotate(inverse,direction)
        projection=local_x@self.axes.T
        center_x=position[...,:1]+local_x@self.centers.T
        radial=torch.where(self.kinds==0,self.radii,
            self.radii*torch.sqrt((1-projection.square()).clamp_min(0.)))
        round_extent=projection.abs()*self.halflengths+radial
        box_extent=local_x.abs()@self.box_halves.T
        extent=torch.where(self.kinds==2,box_extent,round_extent)
        return (center_x+extent).amax(-1)


def string_prism_gap(ball_position, center, quaternion, config, radius):
    """Distance/sign to the same native string surface, including r2 strips."""
    import torch
    from .native import quat_rotate
    inverse = torch.cat((quaternion[..., :1], -quaternion[..., 1:]), -1)
    local = quat_rotate(inverse, ball_position-center)
    if config.get('string_collision_representation')=='inscribed-z-strips':
        centers,half=_strip_tensors(tuple(config['assumptions']['strings_semiaxes_yz_m']),
            config['strings_polygon_sides'],config['assumptions']['string_thickness_m'],
            config['strings_strip_count'],str(local.device),local.dtype)
        delta=(local[...,None,:]-centers).abs()-half
        gap=delta.clamp_min(0).norm(dim=-1)+delta.max(dim=-1).values.clamp_max(0)
        return gap.min(-1).values-radius
    polygon = torch.as_tensor(string_polygon(config), dtype=local.dtype, device=local.device)
    start, edge = polygon, polygon.roll(-1,0)-polygon
    delta = local[...,None,1:]-start
    fraction = ((delta*edge).sum(-1)/edge.square().sum(-1)).clamp(0,1)
    distances = (delta-fraction[...,None]*edge).norm(dim=-1)
    inside = (edge[:,0]*delta[...,1]-edge[:,1]*delta[...,0]>=0).all(-1)
    planar = distances.min(-1).values*torch.where(inside,-1.,1.)
    q = torch.stack((planar, local[...,0].abs()-config['assumptions']['string_thickness_m']/2), -1)
    return q.clamp_min(0).norm(dim=-1)+q.max(-1).values.clamp_max(0)-radius


def swept_string_prism(start,end,center_start,center_end,quat_start,quat_end,config,radius,
                       iterations=4096,tolerance=1e-6):
    from .native import swept_sphere_box
    half = [config['assumptions']['string_thickness_m']/2,
            *config['assumptions']['strings_semiaxes_yz_m']]
    def gap(position,center,quaternion,unused_half,sphere_radius):
        return string_prism_gap(position,center,quaternion,config,sphere_radius)
    return swept_sphere_box(start,end,center_start,center_end,quat_start,quat_end,
                             half,radius,iterations,tolerance,gap_function=gap)
