#!/usr/bin/env python3
"""Convert the pinned official standard PiPER URDF without altering inertials."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
launcher = AppLauncher(args)
app = launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from pxr import Usd, UsdPhysics

try:
    vendor = ROOT / "assets/vendor/piper_isaac_sim"
    source = vendor / "piper_description/urdf/piper_no_gripper_description.urdf"
    tree = ET.parse(source)
    robot = tree.getroot()
    masses = {}
    for link in robot.findall("link"):
        inertial = link.find("inertial")
        if inertial is None or inertial.find("mass") is None or inertial.find("inertia") is None:
            raise ValueError(f"Official link is missing inertial data: {link.attrib['name']}")
        masses[link.attrib["name"]] = float(inertial.find("mass").attrib["value"])
    mesh_hashes = {}
    for mesh in robot.iter("mesh"):
        original = mesh.attrib["filename"]
        if not original.startswith("package://piper_description/"):
            raise ValueError(original)
        path = vendor / "piper_description" / original.removeprefix("package://piper_description/")
        if not path.is_file():
            raise FileNotFoundError(path)
        mesh_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        mesh.set("filename", str(path))
    resolved = args.output / "piper_resolved.urdf"
    tree.write(resolved, encoding="utf-8", xml_declaration=True)
    converted = UrdfConverter(UrdfConverterCfg(
        asset_path=str(resolved.resolve()), usd_dir=str((args.output / "usd").resolve()),
        usd_file_name="piper_standard.usd", fix_base=False, merge_fixed_joints=False,
        force_usd_conversion=True, make_instanceable=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=80.0, damping=4.0)),
    ))
    stage = Usd.Stage.Open(converted.usd_path)
    links = {}
    roots = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            roots.append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            api = UsdPhysics.MassAPI(prim)
            links[prim.GetName()] = {"path": str(prim.GetPath()), "mass": api.GetMassAttr().Get(),
                                     "com": list(api.GetCenterOfMassAttr().Get()),
                                     "diagonal_inertia": list(api.GetDiagonalInertiaAttr().Get())}
    for name, mass in masses.items():
        if name not in links or abs(links[name]["mass"] - mass) > 1e-5:
            raise ValueError(f"Converted mass mismatch: {name}")
    result = {"source": str(source.relative_to(ROOT)),
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "vendor_commit": subprocess.check_output(["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True).strip(),
              "vendor_url": "https://github.com/agilexrobotics/piper_isaac_sim",
              "variant": "standard PiPER, official no-gripper URDF", "mesh_hashes": mesh_hashes,
              "source_masses": masses, "total_arm_mass_kg": sum(masses.values()),
              "usd_path": converted.usd_path, "articulation_roots": roots, "converted_links": links,
              "mass_match": True, "edits_to_urdf": "mesh URI resolution only"}
    (args.output / "conversion.json").write_text(json.dumps(result, indent=2) + "\n")
    print("PIPER_CONVERSION=" + json.dumps({k: result[k] for k in ("mass_match", "total_arm_mass_kg", "usd_path", "articulation_roots")}), flush=True)
except BaseException:
    error = traceback.format_exc()
    (args.output / "error.txt").write_text(error)
    print(error, file=sys.stderr, flush=True)
    raise
finally:
    app.close()
