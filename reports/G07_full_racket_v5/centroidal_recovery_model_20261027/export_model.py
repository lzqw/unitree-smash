"""Read the composed physical USD into a known-model CPU description."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from pxr import Usd, UsdGeom, UsdPhysics

HERE = Path(__file__).resolve().parent
MAIN = HERE.parents[2]
ROOT = MAIN.parent / 'unitree_smash_contact_recovery'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quaternion(value):
    return [float(value.GetReal()), *map(float, value.GetImaginary())]


def main():
    asset = ROOT / 'assets/quadsmash_full_racket_v4_r3.usd'
    geometry = HERE.parent / 'measured_swing9_n128_01/geometry.json'
    names = json.loads(geometry.read_text())['body_names']
    stage = Usd.Stage.Open(str(asset))
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.
    mass_unit = UsdPhysics.GetStageKilogramsPerUnit(stage)
    assert mass_unit == 1.
    bodies = {}
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        name = prim.GetName()
        assert name not in bodies
        api = UsdPhysics.MassAPI(prim)
        bodies[name] = dict(name=name, path=str(prim.GetPath()), mass_kg=float(api.GetMassAttr().Get()),
            center_of_mass_m=list(map(float, api.GetCenterOfMassAttr().Get())),
            diagonal_inertia_kg_m2=list(map(float, api.GetDiagonalInertiaAttr().Get())),
            principal_axes_wxyz=quaternion(api.GetPrincipalAxesAttr().Get()))
    assert set(bodies) == set(names) and len(names) == 28
    lookup = {row['path']: row['name'] for row in bodies.values()}
    joints = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        api = UsdPhysics.Joint(prim)
        parent, child = api.GetBody0Rel().GetTargets(), api.GetBody1Rel().GetTargets()
        assert len(parent) == len(child) == 1
        assert str(parent[0]) in lookup and str(child[0]) in lookup
        revolute = prim.IsA(UsdPhysics.RevoluteJoint)
        assert revolute or prim.IsA(UsdPhysics.FixedJoint)
        joints.append(dict(name=prim.GetName(), kind='revolute' if revolute else 'fixed',
            parent=lookup[str(parent[0])], child=lookup[str(child[0])],
            parent_position_m=list(map(float, api.GetLocalPos0Attr().Get())),
            child_position_m=list(map(float, api.GetLocalPos1Attr().Get())),
            parent_rotation_wxyz=quaternion(api.GetLocalRot0Attr().Get()),
            child_rotation_wxyz=quaternion(api.GetLocalRot1Attr().Get()),
            axis=(str(UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get()) if revolute else None)))
    canonical = [f'{leg}_{part}_joint' for part in ['hip', 'thigh', 'calf'] for leg in ['FL', 'FR', 'RL', 'RR']]
    canonical += [f'joint{i}' for i in range(1, 7)]
    assert {j['name'] for j in joints if j['kind'] == 'revolute'} == set(canonical)
    assert len(joints) == len(names)-1
    assert set(names)-{j['child'] for j in joints} == {'base'}
    dependencies = {str(Path(layer.realPath)): sha(Path(layer.realPath))
                    for layer in stage.GetUsedLayers() if layer.realPath}
    dependencies.update({str(geometry): sha(geometry), str(Path(__file__)): sha(Path(__file__))})
    record = dict(exported_at=datetime.now(timezone.utc).isoformat(), asset_sha256=sha(asset),
        root_body='base', body_names=names, joint_order=canonical,
        bodies=[bodies[name] for name in names], joints=joints,
        stage_meters_per_unit=1., stage_kilograms_per_unit=mass_unit,
        total_authored_mass_kg=sum(b['mass_kg'] for b in bodies.values()),
        source_sha256=dependencies, freshly_read_native_PhysX_mass_tensors=False,
        native_execution=False, formal_acceptance=False)
    with (HERE / 'MODEL.json').open('x') as stream:
        stream.write(json.dumps(record, indent=2)+'\n')
    print(json.dumps(dict(bodies=len(names), joints=len(joints),
        total_authored_mass_kg=record['total_authored_mass_kg'], asset_sha256=record['asset_sha256'])))


if __name__ == '__main__':
    main()
