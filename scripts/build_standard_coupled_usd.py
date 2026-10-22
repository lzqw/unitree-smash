#!/usr/bin/env python3
"""Assemble a source-traceable standard PiPER with explicit mount/tool payloads."""
import argparse
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
launcher = AppLauncher(args)
app = launcher.app
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


def fixed_joint(stage, name, parent, child, translation):
    joint = UsdPhysics.FixedJoint.Define(stage, f"/World/{name}")
    joint.CreateBody0Rel().SetTargets([Sdf.Path(parent)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(child)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*translation))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))


def payload(stage, name, size, mass):
    # The rigid body frame is the payload COM; its collider is centered there.
    body = UsdGeom.Xform.Define(stage, f"/World/{name}").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    api = UsdPhysics.MassAPI.Apply(body)
    api.CreateMassAttr(mass)
    api.CreateCenterOfMassAttr(Gf.Vec3f(0.0))
    x, y, z = size
    api.CreateDiagonalInertiaAttr(Gf.Vec3f(mass * (y*y+z*z)/12, mass * (x*x+z*z)/12, mass * (x*x+y*y)/12))
    api.CreatePrincipalAxesAttr(Gf.Quatf(1.0))
    shape = UsdGeom.Cube.Define(stage, f"/World/{name}/Collider")
    shape.CreateSizeAttr(1.0)
    shape.AddScaleOp().Set(Gf.Vec3d(*size))
    UsdPhysics.CollisionAPI.Apply(shape.GetPrim())


try:
    cfg = json.loads((ROOT / "configs/assembly_v2.json").read_text())
    piper = ROOT / "assets/robots/piper_standard/piper_standard.usd"
    out = args.output / "robot.usd"
    stage = Usd.Stage.CreateNew(str(out))
    world = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(world)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.DefinePrim("/World/Go2", "Xform").GetReferences().AddReference("/home/autodrive/Isaac/unitree_model/Go2/usd/go2.usd")
    stage.DefinePrim("/World/Piper", "Xform").GetReferences().AddReference(str(piper))
    stage.Load()
    arm_base = stage.GetPrimAtPath("/World/Piper/base_link")
    if not arm_base:
        raise RuntimeError("Converted standard PiPER base_link is missing")
    arm_base.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    m, r = cfg["mount"], cfg["racket"]
    payload(stage, "MountPlate", m["plate_size_m"], m["plate_mass_kg"])
    payload(stage, "Racket", r["blade_size_m"], r["mass_kg"])
    fixed_joint(stage, "PlateMount", "/World/Go2/base", "/World/MountPlate", m["plate_center_relative_to_go2_base_m"])
    fixed_joint(stage, "MountJoint", "/World/MountPlate", "/World/Piper/base_link", m["arm_base_relative_to_plate_m"])
    fixed_joint(stage, "RacketMount", "/World/Piper/link6", "/World/Racket", r["blade_center_relative_to_link6_m"])
    static_colliders = []
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            parent = prim
            while parent and not parent.HasAPI(UsdPhysics.RigidBodyAPI):
                parent = parent.GetParent()
            if not parent:
                static_colliders.append(str(prim.GetPath()))
    if static_colliders:
        raise RuntimeError(f"Robot asset contains standalone scene colliders: {static_colliders}")
    stage.GetRootLayer().Save()
    result = {"usd_path": str(out.resolve()), "static_colliders": static_colliders, "assembly": cfg,
              "articulation_roots": [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]}
    (args.output / "assembly.json").write_text(json.dumps(result, indent=2) + "\n")
    print("STANDARD_ASSEMBLY=" + json.dumps({k:result[k] for k in ("usd_path", "static_colliders", "articulation_roots")}), flush=True)
except BaseException:
    error = traceback.format_exc()
    (args.output / "error.txt").write_text(error)
    print(error, file=sys.stderr, flush=True)
    raise
finally:
    app.close()
