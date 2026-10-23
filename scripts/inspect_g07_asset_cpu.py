#!/usr/bin/env python3
"""Inspect USD composition on CPU without starting Kit, PhysX or a GPU app."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        from pxr import Usd
    except ImportError:
        if os.environ.get('QUADSMASH_USD_CPU_BOOTSTRAP'):
            raise
        packages = Path(sys.prefix) / 'lib/python3.11/site-packages/isaacsim/extscache'
        usd_libs = next(packages.glob('omni.usd.libs-*'))
        env = dict(os.environ)
        env['PYTHONPATH'] = str(usd_libs) + os.pathsep + env.get('PYTHONPATH', '')
        env['LD_LIBRARY_PATH'] = os.pathsep.join((str(usd_libs/'bin'), str(Path(sys.prefix)/'lib'),
                                                env.get('LD_LIBRARY_PATH', '')))
        env['QUADSMASH_USD_CPU_BOOTSTRAP'] = '1'
        os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env)
    from pxr import Usd
    path = ROOT/'assets/quadsmash_standard_v2.usd'
    stage = Usd.Stage.Open(str(path))
    def attr(prim, key):
        return stage.GetPrimAtPath(prim).GetAttribute(key).Get()
    collider = '/World/Racket/Collider'
    size = float(attr(collider, 'size'))
    dimensions = [size * float(x) for x in attr(collider, 'xformOp:scale')]
    mount = '/World/RacketMount'
    joint = stage.GetPrimAtPath(mount)
    result = dict(
        asset=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        inspection='USD composition only; no simulator or GPU initialized',
        blade_size_m=dimensions,
        mass_kg=float(attr('/World/Racket', 'physics:mass')),
        diagonal_inertia_kg_m2=list(attr('/World/Racket', 'physics:diagonalInertia')),
        mount_body0=[str(p) for p in joint.GetRelationship('physics:body0').GetTargets()],
        mount_body1=[str(p) for p in joint.GetRelationship('physics:body1').GetTargets()],
        mount_translation_m=list(attr(mount, 'physics:localPos0')),
        mount_rotation_wxyz=[float(attr(mount, 'physics:localRot0').GetReal()),
                              *list(attr(mount, 'physics:localRot0').GetImaginary())],
        provenance=dict(standard_arm='pinned vendor URDF and standard_piper_manifest.json',
                        racket='parameterized placeholder, not measured manufacturer data',
                        tool_mount='assembly_v2.json assumption, not a confirmed physical grasp'),
        metadata_discrepancy='standard_racket_v1.json describes an old plane model/zero mount; current USD uses a centered box 0.18 m from link6',
        physical_design_verified=False,
        physical_dimensions_changed=False,
        limit='No local measured racket/handle/CAD specification establishes that this placeholder equals the intended physical racket.')
    assembly = json.loads((ROOT/'configs/assembly_v2.json').read_text())
    assert all(abs(x-y) < 1e-7 for x,y in zip(dimensions, assembly['racket']['blade_size_m']))
    assert abs(result['mass_kg'] - assembly['racket']['mass_kg']) < 1e-7
    assert result['mount_body0'] == ['/World/Piper/link6']
    assert all(abs(x-y) < 1e-7 for x,y in zip(result['mount_translation_m'],
                                           assembly['racket']['blade_center_relative_to_link6_m']))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
