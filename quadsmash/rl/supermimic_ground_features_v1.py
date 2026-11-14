"""Batched authored nonfoot surface heights for physical-model reward features.

Analytic primitives / authored mesh hulls, NOT cooked PhysX shapes, contact
forces, or safety certificates. Native task contact labels remain authoritative.
"""
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

GEOMETRY = Path(__file__).resolve().parents[2] / 'configs/go2_piper_authored_ground_geometry_v1.json'


class AuthoredGroundFeaturesV1(nn.Module):
    def __init__(self, body_names, asset_sha256, *, geometry_path=GEOMETRY, dtype=torch.float32):
        super().__init__()
        path = Path(geometry_path)
        record = json.loads(path.read_text())
        if asset_sha256 not in record['asset_sha256'].values():
            raise ValueError('Ground geometry does not name the current asset')
        # Check the exported primitive source dependencies, not just the top USD.
        for name, expected in record['asset_sha256'].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
                raise ValueError('Ground geometry source changed: ' + name)
        ids = {name: i for i,name in enumerate(body_names)}
        shapes = record['shapes']
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.paths = tuple(s['path'] for s in shapes)
        self.bodies = tuple(s['body'] for s in shapes)
        self.register_buffer('body_ids',torch.tensor([ids[s['body']] for s in shapes],dtype=torch.long))
        transforms = torch.tensor([s['transform'] for s in shapes],dtype=dtype)
        self.register_buffer('local_rotation',transforms[:,:3,:3].transpose(-1,-2))
        self.register_buffer('local_position',transforms[:,3,:3])
        self.register_buffer('radius',torch.tensor([s.get('radius',0.) for s in shapes],dtype=dtype))
        self.register_buffer('height',torch.tensor([s.get('height',0.) for s in shapes],dtype=dtype))
        self.register_buffer('size',torch.tensor([s.get('size',0.) for s in shapes],dtype=dtype))
        axis = torch.tensor([s.get('axis',0) for s in shapes],dtype=torch.long)
        self.register_buffer('axis_mask',torch.nn.functional.one_hot(axis,3).to(dtype))
        kinds = ('Cube','Sphere','Cylinder','Capsule','Mesh')
        if any(s['kind'] not in kinds for s in shapes):
            raise ValueError('Unsupported authored collider')
        self.register_buffer('kind',torch.tensor([kinds.index(s['kind']) for s in shapes]))
        mesh_ids = [i for i,s in enumerate(shapes) if s['kind']=='Mesh']
        self.register_buffer('mesh_ids',torch.tensor(mesh_ids,dtype=torch.long))
        max_points = max(len(shapes[i]['points']) for i in mesh_ids)
        points = torch.zeros(len(mesh_ids),max_points,3,dtype=dtype)
        valid = torch.zeros(len(mesh_ids),max_points,dtype=torch.bool)
        for k,i in enumerate(mesh_ids):
            count=len(shapes[i]['points'])
            points[k,:count]=torch.tensor(shapes[i]['points'],dtype=dtype)
            valid[k,:count]=True
        self.register_buffer('mesh_points',points)
        self.register_buffer('mesh_valid',valid)
        self.register_buffer('head_ids',torch.tensor([i for i,s in enumerate(shapes) if s['body'] in ('Head_upper','Head_lower')]))

    def forward(self, positions, rotations, ground_z=0.):
        p, r = positions[:,self.body_ids],rotations[:,self.body_ids]
        # USD row-vector local transforms are transposed during construction.
        direction = (r @ self.local_rotation)[...,2,:]
        center_z = p[...,2] + (r[...,2,:]*self.local_position).sum(-1)
        norm = direction.norm(dim=-1)
        axial = (direction*self.axis_mask).sum(-1).abs()
        radial = (direction*(1-self.axis_mask)).norm(dim=-1)
        cube = -.5*self.size*direction.abs().sum(-1)
        sphere = -self.radius*norm
        cylinder = -self.radius*radial-.5*self.height*axial
        capsule = sphere-.5*self.height*axial
        support = torch.where(self.kind==0,cube,torch.where(self.kind==1,sphere,
                    torch.where(self.kind==2,cylinder,capsule)))
        projection = torch.einsum('bsi,svi->bsv',direction[:,self.mesh_ids],self.mesh_points)
        mesh_min = projection.masked_fill(~self.mesh_valid,float('inf')).amin(-1)
        support = support.index_copy(1,self.mesh_ids,mesh_min)
        heights = center_z + support - ground_z
        return dict(shape_height=heights,head_height=heights[:,self.head_ids].amin(1),
                    nonfoot_height=heights.amin(1))
