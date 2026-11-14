"""Causal, differentiable contact-support preview for learned ball responses.

Original compound geometry is reused. Motion over one control tick is only
constant relative velocity for the racket and ballistic for ground/net; this
is not PhysX contact certification. No native physics or success rule changes.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from quadsmash.physics.full_racket_v4 import parts,string_strips
from quadsmash.physics.court_v4 import net_band_segments


def box_gap(points, centers, halves):
    delta=(points[...,None,:]-centers).abs()-halves
    return delta.clamp_min(0).norm(dim=-1)+delta.amax(-1).clamp_max(0)


def round_gap(points, centers, axes, halves, radii, capsule):
    delta=points[...,None,:]-centers
    axial=(delta*axes).sum(-1)
    radial=(delta-axial[...,None]*axes).norm(dim=-1)
    capped=(axial.abs()-halves).clamp_min(0)
    cap_gap=torch.stack((radial,capped),-1).norm(dim=-1)-radii
    q=torch.stack((radial-radii,axial.abs()-halves),-1)
    cylinder_gap=q.clamp_min(0).norm(dim=-1)+q.amax(-1).clamp_max(0)
    return torch.where(capsule,cap_gap,cylinder_gap)


class BallGeometryPreviewV1(nn.Module):
    schema='native-geometry-causal-ball-preview-v1'
    def __init__(self,root=None,softness_m=.001):
        super().__init__()
        root=Path(__file__).resolve().parents[2] if root is None else Path(root)
        files=['configs/full_racket_v4_r3.json','configs/court_net_v4.json','configs/shuttle_physics_v2.json']
        self.geometry_hashes={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files}
        racket,court,ball=[json.loads((root/f).read_text()) for f in files]
        self.radius=float(ball['collision_radius_m']);self.softness_m=float(softness_m)
        if self.softness_m<=0:raise ValueError('Positive surrogate transition width required')
        def register(name,data,dtype=torch.float32):self.register_buffer(name,torch.as_tensor(np.asarray(data),dtype=dtype))
        components=[x for x in parts(racket) if x['collision']]
        boxes=[(s['center'],np.array(s['size'])/2) for s in string_strips(racket)]
        boxes += [(p['center'],p['size']/2) for p in components if p['kind']=='box']
        register('racket_box_centers',[x[0] for x in boxes]);register('racket_box_halves',[x[1] for x in boxes])
        rounds=[x for x in components if x['kind'] in ('capsule','cylinder')]
        register('racket_round_centers',[x['center'] for x in rounds]);register('racket_round_axes',[x['axis'] for x in rounds])
        register('racket_round_halves',[x['length']/2 for x in rounds]);register('racket_round_radii',[x['radius'] for x in rounds])
        register('racket_round_capsule',[x['kind']=='capsule' for x in rounds],torch.bool)
        net=[]
        for points,_ in net_band_segments(court):
            xyz=np.asarray(points);center=(xyz.min(0)+xyz.max(0))/2;center[0]+=court['net_plane_x_m']
            net.append((center,(xyz.max(0)-xyz.min(0))/2))
        register('net_centers',[x[0] for x in net]);register('net_halves',[x[1] for x in net])
        register('post_centers',[[court['net_plane_x_m'],sign*court['post_y_center_abs_m'],court['post_height_m']/2] for sign in (-1,1)])
        register('post_axes',[[0,0,1],[0,0,1]])
        register('post_halves',[court['post_height_m']/2]*2);register('post_radii',[court['post_radius_m']]*2)
        register('post_capsule',[False,False],torch.bool)
        # Same 0.01 s / 62.5 us clock; motion prediction is explicitly approximate.
        control_dt=float(ball['policy_dt_s']);physics_dt=float(ball['native_physics_dt_s'])
        count=round(control_dt/physics_dt)
        if abs(count*physics_dt-control_dt)>1e-12:raise ValueError('Nonintegral substep clock')
        register('times',np.linspace(0.,control_dt,count+1))

    def racket_gap(self,local):
        b=box_gap(local,self.racket_box_centers,self.racket_box_halves).amin(-1)
        r=round_gap(local,self.racket_round_centers,self.racket_round_axes,
                    self.racket_round_halves,self.racket_round_radii,self.racket_round_capsule).amin(-1)
        return torch.minimum(b,r)-self.radius

    def net_gap(self,world):
        b=box_gap(world,self.net_centers,self.net_halves).amin(-1)
        r=round_gap(world,self.post_centers,self.post_axes,self.post_halves,self.post_radii,self.post_capsule).amin(-1)
        return torch.minimum(b,r)-self.radius

    def forward(self,features):
        if len(features)>256:
            chunks=[self(part) for part in features.split(256)]
            return tuple(torch.cat([x[i] for x in chunks]) for i in range(2))
        t=self.times[None,:,None]
        local=features[:,None,6:9]+t*features[:,None,9:12]
        world=features[:,None,:3]+t*features[:,None,3:6]
        world=world+world.new_tensor([0.,0.,-4.905])*t.square()
        gap=torch.stack((self.racket_gap(local).amin(-1),
                         (world[:,:,2]-self.radius).amin(-1),self.net_gap(world).amin(-1)),-1)
        # Far-away corrections vanish smoothly. This width is a model setting,
        # not an enlargement of native collision shapes or scoring thresholds.
        return (-gap/self.softness_m).sigmoid(),gap


class GeometrySupportedBallV1(nn.Module):
    def __init__(self,ball,preview):
        super().__init__();self.ball=ball;self.preview=preview

    def components(self,features):
        p,response=self.ball.components(features);support,_=self.preview(features)
        weighted=p[:,1:]*torch.cat((support,torch.ones_like(support[:,:1])),-1)
        # Removed mass returns to free flight; no renormalization inflates other modes.
        adjusted=torch.cat((1.-weighted.sum(-1,keepdim=True),weighted),-1)
        return adjusted,response

    def forward(self,features,present):
        p,response=self.components(features)
        return present[:,None].to(response.dtype)*(p[:,1:,None]*response).sum(1),1.-p[:,0]
