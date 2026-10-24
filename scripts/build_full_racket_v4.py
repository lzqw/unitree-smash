#!/usr/bin/env python3
"""Version the legacy robot with a finite-mass full racket and grip fixture."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Gf,Sdf,Usd,UsdGeom,UsdPhysics,PhysxSchema
from quadsmash.physics.full_racket_v4 import load_racket,parts,mass_properties,string_prism_mesh,string_strips


def quat(matrix):
    x,y,z,w = Rotation.from_matrix(matrix).as_quat()
    return Gf.Quatf(float(w),Gf.Vec3f(float(x),float(y),float(z)))


def main():
    args.output.mkdir(parents=True,exist_ok=True)
    cfg = load_racket()
    source, output = ROOT/cfg['legacy_asset'], ROOT/cfg['output_asset']
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if output.exists():
        raise ValueError('Never overwrite an existing versioned asset')
    stage = Usd.Stage.Open(str(source))
    stage.GetRootLayer().Export(str(output))
    stage = Usd.Stage.Open(str(output))
    if not stage.GetPrimAtPath('/World/Racket'):
        raise ValueError('Legacy tool body missing')
    for prim in list(stage.Traverse()):
        if prim.IsA(UsdPhysics.FixedJoint):
            targets = UsdPhysics.FixedJoint(prim).GetBody1Rel().GetTargets()
            if Sdf.Path('/World/Racket') in targets:
                stage.RemovePrim(prim.GetPath())
    stage.RemovePrim('/World/Racket')
    body = UsdGeom.Xform.Define(stage,'/World/Racket').GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    properties = mass_properties(cfg)
    mass = UsdPhysics.MassAPI.Apply(body)
    mass.CreateMassAttr(properties['mass_kg'])
    mass.CreateCenterOfMassAttr(Gf.Vec3f(*properties['center_of_mass_in_body_m']))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*properties['principal_inertia_kg_m2']))
    mass.CreatePrincipalAxesAttr(quat(properties['principal_axes_in_body']))
    colors = dict(Handle=(.40,.25,.12),Grip=(.12,.13,.15),Shaft=(.10,.14,.20),Clamp=(.42,.45,.49))
    for part in parts(cfg):
        path = '/World/Racket/'+part['name']
        kind = part['kind']
        strips = kind=='string_prism' and cfg.get('string_collision_representation')=='inscribed-z-strips'
        if strips:
            shape=UsdGeom.Xform.Define(stage,path)
            for index,strip in enumerate(string_strips(cfg)):
                box=UsdGeom.Cube.Define(stage,f'{path}/strip_{index:02d}')
                box.CreateSizeAttr(1.)
                box.AddTranslateOp().Set(Gf.Vec3d(*strip['center']))
                box.AddScaleOp().Set(Gf.Vec3d(*strip['size']))
                box.CreateDisplayColorAttr([Gf.Vec3f(.65,.78,.79)])
                UsdPhysics.CollisionAPI.Apply(box.GetPrim())
                api=PhysxSchema.PhysxCollisionAPI.Apply(box.GetPrim())
                api.CreateContactOffsetAttr(.0001);api.CreateRestOffsetAttr(0.)
                box.GetPrim().SetCustomDataByKey('quadsmashComponent','Strings')
        elif kind == 'string_prism':
            shape = UsdGeom.Mesh.Define(stage,path)
            points,faces = string_prism_mesh(cfg)
            shape.CreatePointsAttr(points)
            shape.CreateFaceVertexCountsAttr([len(face) for face in faces])
            shape.CreateFaceVertexIndicesAttr([i for face in faces for i in face])
            shape.CreateSubdivisionSchemeAttr('none')
            UsdPhysics.MeshCollisionAPI.Apply(shape.GetPrim()).CreateApproximationAttr('convexHull')
        elif kind == 'box':
            shape = UsdGeom.Cube.Define(stage,path)
            shape.CreateSizeAttr(1.)
            shape.AddTranslateOp().Set(Gf.Vec3d(*part['center']))
            shape.AddScaleOp().Set(Gf.Vec3d(*part['size']))
        else:
            shape = (UsdGeom.Capsule if kind=='capsule' else UsdGeom.Cylinder).Define(stage,path)
            shape.CreateRadiusAttr(part['radius'])
            shape.CreateHeightAttr(part['length'])
            shape.CreateAxisAttr('Z')
            shape.AddTranslateOp().Set(Gf.Vec3d(*part['center']))
            shape.AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0,0,1),Gf.Vec3d(*part['axis'])).GetQuat()))
        color = colors.get(part['name'], (.10,.35,.55) if kind=='capsule' else (.65,.78,.79))
        if not strips:shape.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        if part['collision'] and not strips:
            UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
            api = PhysxSchema.PhysxCollisionAPI.Apply(shape.GetPrim())
            api.CreateContactOffsetAttr(.0001)
            api.CreateRestOffsetAttr(0.)
        shape.GetPrim().SetCustomDataByKey('quadsmashComponent',part['name'])
    joint = UsdPhysics.FixedJoint.Define(stage,'/World/RacketMount')
    joint.CreateBody0Rel().SetTargets([Sdf.Path('/World/Piper/link6')])
    joint.CreateBody1Rel().SetTargets([Sdf.Path('/World/Racket')])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*cfg['string_center_in_flange_m']))
    w,x,y,z = cfg['body_rotation_in_flange_wxyz']
    joint.CreateLocalRot0Attr(Gf.Quatf(w,Gf.Vec3f(x,y,z)))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0,0,0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.))
    body.SetCustomDataByKey('quadsmashModelVersion',cfg['version'])
    stage.GetRootLayer().Save()
    assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash
    result = dict(asset=str(output.relative_to(ROOT)),asset_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                  legacy_asset_sha256=source_hash,configuration=cfg,mass_properties=properties,
                  old_passes_inherited=False,physical_validation_pending=True)
    (args.output/'asset_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.output/'completed.json').write_text(json.dumps(dict(asset=result['asset'],source_preserved=True))+'\n')
    print(json.dumps(result),flush=True)


try:
    main()
except BaseException:
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'error.txt').write_text(traceback.format_exc())
    raise
finally:
    app.close(skip_cleanup=True)
