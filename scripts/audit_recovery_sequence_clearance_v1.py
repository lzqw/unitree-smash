#!/usr/bin/env python3
"""Check authored ground clearance along model plans, not cooked PhysX contacts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve clearance evidence')
    try:from pxr import Usd,UsdGeom,UsdPhysics
    except ImportError:
        if os.environ.get('RECOVERY_CLEARANCE_BOOTSTRAP'):raise
        libraries=next((Path(sys.prefix)/'lib/python3.11/site-packages/isaacsim/extscache').glob('omni.usd.libs-*'))
        env=dict(os.environ,RECOVERY_CLEARANCE_BOOTSTRAP='1')
        env['PYTHONPATH']=str(libraries)+os.pathsep+env.get('PYTHONPATH','')
        env['LD_LIBRARY_PATH']=os.pathsep.join([str(libraries/'bin'),str(Path(sys.prefix)/'lib'),env.get('LD_LIBRARY_PATH','')])
        os.execve(sys.executable,[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]],env)
    import numpy as np
    from scripts.parallel_landing_solver_v1 import build_controller
    from scripts.loaded_pinocchio_model_v1 import rotation_wxyz
    manifest=json.loads((a.bundle/'manifest.json').read_text())
    archive=a.bundle/manifest['bundle_name']
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=manifest['bundle_sha256']:raise ValueError('Plan changed')
    protocol_path=ROOT/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'
    protocol=json.loads(protocol_path.read_text());asset=ROOT/protocol['asset']['path']
    if hashlib.sha256(asset.read_bytes()).hexdigest()!=protocol['asset']['sha256']:raise ValueError('Asset changed')
    stage=Usd.Stage.Open(str(asset));cache=UsdGeom.XformCache();shapes=[]
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        owner=prim
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            if prim.GetTypeName()!='Mesh':continue
            owner=prim.GetParent()
            while owner and not owner.HasAPI(UsdPhysics.CollisionAPI):
                if owner.HasAPI(UsdPhysics.RigidBodyAPI):break
                owner=owner.GetParent()
        if not owner or not owner.HasAPI(UsdPhysics.CollisionAPI) or owner.GetAttribute('physics:collisionEnabled').Get() is False:continue
        kind=prim.GetTypeName()
        if kind=='Xform':continue
        if kind not in ['Sphere','Cylinder','Capsule','Cube','Mesh']:raise ValueError('Unknown collider '+kind)
        body=prim
        while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):body=body.GetParent()
        if not body:raise ValueError('Missing collider body')
        name=body.GetName()
        if name.endswith('_foot'):continue
        transform=np.array(cache.GetLocalToWorldTransform(prim)*cache.GetLocalToWorldTransform(body).GetInverse())
        shape=dict(body=name,path=str(prim.GetPath()),kind=kind,transform=transform)
        if kind=='Mesh':shape['points']=np.asarray(prim.GetAttribute('points').Get(),dtype=float)
        if kind=='Cube':shape['size']=float(prim.GetAttribute('size').Get())
        if kind in ['Sphere','Cylinder','Capsule']:shape['radius']=float(prim.GetAttribute('radius').Get())
        if kind in ['Cylinder','Capsule']:
            shape.update(height=float(prim.GetAttribute('height').Get()),axis='XYZ'.index(prim.GetAttribute('axis').Get()))
        shapes.append(shape)
    controller=build_controller(manifest['configuration']);model=controller.qp.model;rows=[]
    baseline=Path(manifest['baseline'])
    geometry=json.loads((baseline/'geometry.json').read_text())
    if geometry['body_names']!=model.names:raise ValueError('Native/model body order differs')
    with np.load(baseline/'all_first_episodes.npz',allow_pickle=False) as actual:
        measured_poses=actual['link_poses'].copy();valid=actual['valid_first_episode'].copy()
    with np.load(archive,allow_pickle=False) as data:
        for item in manifest['rows']:
            key=f"case_{item['case']:03d}";states=data[key+'_predicted_q'];velocities=data[key+'_predicted_v'];frames=[]
            for k,(q,v) in enumerate(zip(states,velocities)):
                snap=model.snapshot(q,v);minimum=[]
                for shape in shapes:
                    i=model.names.index(shape['body']);R=snap['body_rotations_world'][i];T=shape['transform']
                    W=R@T[:3,:3].T;center=R@T[3,:3]+snap['body_positions_world'][i];kind=shape['kind']
                    if kind=='Mesh':lower=float((shape['points']@W[2]+center[2]).min())
                    else:
                        if kind=='Cube':support=.5*shape['size']*abs(W[2]).sum()
                        elif kind=='Sphere':support=shape['radius']*np.linalg.norm(W[2])
                        else:
                            axis=shape['axis'];radial=np.delete(W,axis,axis=1) if kind=='Cylinder' else W
                            support=.5*shape['height']*abs(W[2,axis])+shape['radius']*np.linalg.norm(radial[2])
                        lower=float(center[2]-support)
                    minimum.append(dict(body=shape['body'],path=shape['path'],minimum_z_m=lower))
                frames.append(dict(offset_s=.01*k,nearest=sorted(minimum,key=lambda x:x['minimum_z_m'])[:3]))
            tick=item['query_tick'];case=item['case']
            if tick%2 or not valid[tick//2,case]:raise ValueError('No measured query body poses')
            query=model.snapshot(states[0],velocities[0]);measured=measured_poses[tick//2,case]
            pe=float(np.max(abs(query['body_positions_world']-measured[:,:3])))
            re=float(np.max(abs(query['body_rotations_world']-np.stack([rotation_wxyz(p[3:]) for p in measured]))))
            if pe>2e-5 or re>2e-5:raise ValueError('Loaded geometry does not match actual query poses')
            worst=min(frames,key=lambda x:x['nearest'][0]['minimum_z_m'])
            rows.append(dict(case=case,query_tick=tick,worst=worst,frames=frames,
                actual_query_body_position_error_max_m=pe,actual_query_body_rotation_error_max=re))
    paths=[Path(__file__),a.bundle/'manifest.json',archive,protocol_path,asset,baseline/'geometry.json',baseline/'all_first_episodes.npz',ROOT/'scripts/loaded_pinocchio_model_v1.py']
    layers=json.loads((Path(manifest['baseline'])/'asset_dependencies.json').read_text())['resolved_layers']
    paths += [Path(x) for x in layers]
    result=dict(scope=__doc__,native_execution=False,ground_plane_z_m=0.,
        nonfoot_authored_shapes=len(shapes),cases=rows,
        limitations=['10 ms endpoints only; no substep or self-collision check.',
            'Authored vertices/primitives, not cooked PhysX collision shapes or contact offsets.',
            'No task outcome relabeling; this identifies possible missing geometric constraints.'],
        source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([dict(case=r['case'],worst=r['worst']) for r in rows]))


if __name__=='__main__':main()
