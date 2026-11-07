"""Export unchanged USD collision shapes for offline trajectory constraints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve geometry record')
    try:from pxr import Usd
    except ImportError:
        if os.environ.get('RECOVERY_GEOMETRY_BOOTSTRAP'):raise
        libs=next((Path(sys.prefix)/'lib/python3.11/site-packages/isaacsim/extscache').glob('omni.usd.libs-*'))
        env=dict(os.environ,RECOVERY_GEOMETRY_BOOTSTRAP='1')
        env['PYTHONPATH']=str(libs)+os.pathsep+env.get('PYTHONPATH','')
        env['LD_LIBRARY_PATH']=os.pathsep.join([str(libs/'bin'),str(Path(sys.prefix)/'lib'),env.get('LD_LIBRARY_PATH','')])
        os.execve(sys.executable,[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]],env)
    from scripts.authored_ground_clearance_v1 import load_usd_shapes
    protocol=json.loads((ROOT/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json').read_text())
    asset=ROOT/protocol['asset']['path']
    if hashlib.sha256(asset.read_bytes()).hexdigest()!=protocol['asset']['sha256']:raise ValueError('Asset mismatch')
    shapes=load_usd_shapes(asset);stage=Usd.Stage.Open(str(asset))
    # Linear support in any direction depends only on hull vertices. This
    # removes redundant mesh points without changing the support function.
    import numpy as np
    from scipy.spatial import ConvexHull
    original_points=0;hull_points=0
    for shape in shapes:
        if shape['kind']!='Mesh':continue
        points=np.unique(np.asarray(shape['points']),axis=0);original_points+=len(points)
        center=points.mean(0);_,singular,axes=np.linalg.svd(points-center,full_matrices=False)
        rank=int((singular>max(singular[0],1.)*1e-12).sum())
        if rank>=2 and len(points)>rank+1:
            points=points[ConvexHull((points-center)@axes[:rank].T).vertices]
        elif rank==1:
            values=(points-center)@axes[0];points=points[[values.argmin(),values.argmax()]]
        elif rank==0:points=points[:1]
        hull_points+=len(points);shape['points']=points.tolist()
    paths={Path(layer.realPath).resolve() for layer in stage.GetUsedLayers() if layer.realPath}
    paths.add(asset.resolve())
    record=dict(shapes=shapes,mesh_unique_points=original_points,mesh_hull_points=hull_points,
        asset_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths},
        scope='Authored nonfoot ground clearance only; excludes cooked collision and self-collision.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(record)+'\n')
    print(json.dumps(dict(shapes=len(shapes),bytes=a.output.stat().st_size)))

if __name__=='__main__':main()
