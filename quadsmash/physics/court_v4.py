"""Regulation-scale court and suspended equivalent net; legacy net is separate."""
from pathlib import Path
import json
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'regulation-suspended-band-net-v4'


def load_court():
    return json.loads((ROOT/'configs/court_net_v4.json').read_text())


def segment_top(index,config):
    n = config['net_profile_segments']
    if index in (n//2-1,n//2):return config['net_center_top_m']
    if index in (0,n-1):return config['net_side_top_m']
    y = -1+2*(index+.5)/n
    return config['net_center_top_m']+(config['net_side_top_m']-config['net_center_top_m'])*y*y


def net_band_segments(config):
    half = config['net_width_m']/2
    n = config['net_profile_segments']
    thickness = config['net_equivalent_thickness_m']
    segments = []
    for i in range(n):
        left, right = -half+2*half*i/n, -half+2*half*(i+1)/n
        height = segment_top(i,config)
        top = lambda y: height
        points = [(x,y,z) for x in [-thickness/2,thickness/2]
                  for y in [left,right] for z in [top(y)-config['net_depth_m'],top(y)]]
        faces = [tuple(reversed(face)) for face in
                 [(0,2,3,1),(4,5,7,6),(0,1,5,4),(2,6,7,3),(0,4,6,2),(1,3,7,5)]]
        segments.append((points,faces))
    return segments


def top_height(y, config):
    import torch
    n = config['net_profile_segments']
    if torch.is_tensor(y):
        index = (((y/(config['net_width_m']/2)).clamp(-1,1)+1)*(.5*n)).long().clamp(0,n-1)
        return _top_height_tensor(n,config['net_center_top_m'],config['net_side_top_m'],str(y.device),y.dtype)[index]
    import numpy as np
    values = [segment_top(i,config) for i in range(n)]
    index = np.clip(((np.clip(np.asarray(y)/(config['net_width_m']/2),-1,1)+1)*(.5*n)).astype(int),0,n-1)
    return np.asarray(values)[index]


@lru_cache(maxsize=32)
def _top_height_tensor(count,center,side,device,dtype):
    import torch
    config=dict(net_profile_segments=count,net_center_top_m=center,net_side_top_m=side)
    with torch.inference_mode(False):
        return torch.tensor([segment_top(i,config) for i in range(count)],device=device,dtype=dtype)


def spawn_court_net(stage, path, config):
    from pxr import Gf,Usd,UsdGeom,UsdPhysics,PhysxSchema
    existing = stage.GetPrimAtPath(path)
    if existing and existing.GetCustomDataByKey('quadsmashNetVersion') == VERSION:
        return [str(p.GetPath()) for p in Usd.PrimRange(existing)
                if p.HasAPI(UsdPhysics.CollisionAPI)]
    root = UsdGeom.Xform.Define(stage,path)
    root.AddTranslateOp().Set(Gf.Vec3d(config['net_plane_x_m'],0,0))
    # One immovable source body per environment allows an inverse contact
    # sensor to monitor every robot body and the shuttle without N*B mismatch.
    body = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    body.CreateKinematicEnabledAttr(True)
    PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim()).CreateDisableGravityAttr(True)
    PhysxSchema.PhysxContactReportAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(1.)
    UsdGeom.Xform.Define(stage,path+'/geometry')
    UsdGeom.Xform.Define(stage,path+'/geometry/mesh')
    colliders = []
    for index,(points,faces) in enumerate(net_band_segments(config)):
        mesh = UsdGeom.Cube.Define(stage,f'{path}/geometry/mesh/segment_{index:02d}')
        mesh.CreateSizeAttr(1.)
        left,right = points[0][1],points[2][1]
        bottom,top = points[0][2],points[1][2]
        mesh.AddTranslateOp().Set(Gf.Vec3d(0,(left+right)/2,(top+bottom)/2))
        mesh.AddScaleOp().Set(Gf.Vec3d(config['net_equivalent_thickness_m'],right-left,top-bottom))
        mesh.CreateDisplayColorAttr([Gf.Vec3f(.13,.16,.17)])
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        colliders.append(str(mesh.GetPath()))
        tape_points = [(x,y,z if j%2 else z+config['net_depth_m']-config['net_top_tape_depth_m'])
                       for j,(x,y,z) in enumerate(points)]
        tape = UsdGeom.Mesh.Define(stage,f'{path}/geometry/tape_{index:02d}')
        tape.CreatePointsAttr(tape_points)
        tape.CreateFaceVertexCountsAttr([len(face) for face in faces])
        tape.CreateFaceVertexIndicesAttr([i for face in faces for i in face])
        tape.CreateSubdivisionSchemeAttr('none')
        tape.CreateDisplayColorAttr([Gf.Vec3f(.95,.95,.95)])
    for name,sign in [('PostLeft',-1),('PostRight',1)]:
        post = UsdGeom.Cylinder.Define(stage,path+'/geometry/'+name)
        post.CreateAxisAttr('Z')
        post.CreateRadiusAttr(config['post_radius_m'])
        post.CreateHeightAttr(config['post_height_m'])
        post.AddTranslateOp().Set(Gf.Vec3d(0,sign*config['post_y_center_abs_m'],config['post_height_m']/2))
        post.CreateDisplayColorAttr([Gf.Vec3f(.18,.21,.26)])
        UsdPhysics.CollisionAPI.Apply(post.GetPrim())
        colliders.append(str(post.GetPath()))
    root.GetPrim().SetCustomDataByKey('quadsmashNetVersion',VERSION)
    return colliders


def spawn_court_lines(stage,path,config):
    from pxr import Gf,UsdGeom
    if stage.GetPrimAtPath(path+'/OwnBack').IsValid():
        return
    UsdGeom.Xform.Define(stage,path)
    width,length,line = config['doubles_width_m'],config['court_length_m'],config['line_width_m']
    center = config['net_plane_x_m']
    def mark(name,position,size):
        shape = UsdGeom.Cube.Define(stage,path+'/'+name)
        shape.CreateSizeAttr(1.)
        shape.AddTranslateOp().Set(Gf.Vec3d(*position))
        shape.AddScaleOp().Set(Gf.Vec3d(*size))
        shape.CreateDisplayColorAttr([Gf.Vec3f(.96,.97,.98)])
    for sign,name in [(-1,'Own'),(1,'Opponent')]:
        mark(name+'Back',(center+sign*(length/2-line/2),0,.001),(line,width,.001))
        mark(name+'ShortService',(center+sign*1.98,0,.001),(line,width,.001))
        mark(name+'DoublesLongService',(center+sign*(length/2-.76),0,.001),(line,width,.001))
        mark(name+'CenterService',(center+sign*((length/2+1.98)/2),0,.001),(length/2-1.98,line,.001))
    for sign,name in [(-1,'Left'),(1,'Right')]:
        mark(name+'Doubles',(center,sign*(width/2-line/2),.001),(length,line,.001))
        mark(name+'Singles',(center,sign*(config['singles_width_m']/2-line/2),.001),(length,line,.001))
