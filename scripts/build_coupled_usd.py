#!/usr/bin/env python3
"""Build the local 18-DOF Go2+PiPER USD from pinned source assets."""
from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


ROOT = Path(__file__).resolve().parents[1]
GO2 = Path("/home/autodrive/Isaac/unitree_model/Go2/usd/go2.usd")
PIPER = ROOT / "assets/vendor/piper_isaac_sim/USD/piper_v1.usd"
RACKET_MASS_KG = 0.09
RACKET_HEAD_RADIUS_M = 0.11


def main() -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(args.output))
    world = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(world)
    stage.DefinePrim("/World/Go2", "Xform").GetReferences().AddReference(str(GO2))
    stage.DefinePrim("/World/Piper", "Xform").GetReferences().AddReference(str(PIPER))
    stage.Load()

    # The task policy controls the six arm joints.  The vendor USD's two
    # prismatic gripper joints are excluded from this whole-body contract.
    for path in (
        # The vendor file is a demo scene. Its ground plane is not robot
        # geometry and would otherwise be cloned at the robot spawn height.
        "/World/Piper/GroundPlane",
        "/World/Piper/piper_camera/joints/joint7",
        "/World/Piper/piper_camera/joints/joint8",
        "/World/Piper/piper_camera/link7",
        "/World/Piper/piper_camera/link8",
        "/World/Piper/piper_camera/gripper_base",
        "/World/Piper/piper_camera/root_joint",
    ):
        # Author an explicit root-layer override.  Setting active on a
        # referenced prim alone may only edit the reference layer and is not
        # sufficient for the composed task asset.
        prim = stage.OverridePrim(path)
        prim.SetActive(False)

    mount = UsdPhysics.FixedJoint.Define(stage, Sdf.Path("/World/MountJoint"))
    mount.CreateBody0Rel().SetTargets([Sdf.Path("/World/Go2/base")])
    mount.CreateBody1Rel().SetTargets([Sdf.Path("/World/Piper/piper_camera/arm_base")])
    mount.CreateLocalPos0Attr().Set(Gf.Vec3f(0.15, 0.0, 0.45))
    mount.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    mount.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    mount.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    # The racket is a real coupled rigid body, rather than a policy-only
    # marker.  It is attached to PiPER link6 by a fixed joint, carries the
    # parameterized payload mass from assets/rackets/standard_racket_v1.json,
    # and has an explicit collision shape for later swept-contact tasks.
    racket = UsdGeom.Xform.Define(stage, Sdf.Path("/World/Racket"))
    racket_prim = racket.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(racket_prim)
    mass_api = UsdPhysics.MassAPI.Apply(racket_prim)
    mass_api.CreateMassAttr(RACKET_MASS_KG)
    racket_prim.CreateAttribute("quadsmash:asset_id", Sdf.ValueTypeNames.String).Set("standard_racket_v1")
    racket_prim.CreateAttribute("quadsmash:mass_source", Sdf.ValueTypeNames.String).Set("parameterized_assumption")

    blade = UsdGeom.Cube.Define(stage, Sdf.Path("/World/Racket/Blade"))
    blade.CreateSizeAttr(1.0)
    blade.AddScaleOp().Set(Gf.Vec3f(0.015, RACKET_HEAD_RADIUS_M, RACKET_HEAD_RADIUS_M))
    blade.AddTranslateOp().Set(Gf.Vec3f(0.18, 0.0, 0.0))
    UsdPhysics.CollisionAPI.Apply(blade.GetPrim())

    racket_mount = UsdPhysics.FixedJoint.Define(stage, Sdf.Path("/World/RacketMount"))
    racket_mount.CreateBody0Rel().SetTargets([Sdf.Path("/World/Piper/piper_camera/link6")])
    racket_mount.CreateBody1Rel().SetTargets([Sdf.Path("/World/Racket")])
    racket_mount.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    racket_mount.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    racket_mount.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    racket_mount.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    stage.GetRootLayer().Save()
    print(f"WROTE_COUPLED_USD={args.output}")


try:
    main()
finally:
    app.close()
