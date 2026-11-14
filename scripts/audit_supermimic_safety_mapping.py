#!/usr/bin/env python3
"""Read pinned USD via bundled USD libraries; no AppLauncher or physics step."""
import argparse,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import torch
from pxr import Usd,UsdGeom
from quadsmash.tasks.stand.visual_geometry import collision_proxy_corners


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads((args.archive/'manifest.json').read_text())
    asset=Path(__file__).resolve().parents[1]/manifest['task_contract']['asset']['path']
    digest=hashlib.sha256(asset.read_bytes()).hexdigest()
    if digest!=manifest['task_contract']['asset']['sha256']:raise ValueError('Asset mismatch')
    stage=Usd.Stage.CreateInMemory()
    robot=UsdGeom.Xform.Define(stage,'/World/envs/env_0/Robot').GetPrim()
    robot.GetReferences().AddReference(str(asset))
    ids,points,paths=collision_proxy_corners(stage,SimpleNamespace(body_names=manifest['body_names']))
    errors=[]
    for entry in (manifest['chunks'][0],manifest['chunks'][8],manifest['chunks'][-1]):
        row=torch.load(args.archive/entry['file'],map_location='cpu',weights_only=True)['rows'][0]
        observed=row['after']['auxiliary']['reward_inputs']['safety_body_height']
        reconstructed=row['after']['body_position_w'][:,ids,2]
        torch.testing.assert_close(reconstructed,observed,atol=0,rtol=0)
        errors.append(dict(tick=row['before']['control_tick'],max_height_error=float((reconstructed-observed).abs().max())))
    result=dict(asset_sha256=digest,native_body_ids=ids.tolist(),
        safety_body_names=[manifest['body_names'][i] for i in ids],
        collision_paths=paths,height_checks=errors,
        implementation_sha256=hashlib.sha256(Path('quadsmash/tasks/stand/visual_geometry.py').read_bytes()).hexdigest(),
        native_steps=0)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(body_count=len(ids),height_checks=errors)),flush=True)


if __name__=='__main__':main()
