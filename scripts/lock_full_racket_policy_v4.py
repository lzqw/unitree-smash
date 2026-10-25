#!/usr/bin/env python3
"""Freeze policy, physical inputs and evaluator source before formal panels."""
import argparse,json,subprocess,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from quadsmash.evaluation_v4 import digest
from quadsmash.usd_dependencies_v4 import MANIFEST,verify_dependency_manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--protocol',type=Path,default=ROOT/'configs/g07_full_racket_eval_v4_r3.json')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Lock output must be new')
    protocol=json.loads(a.protocol.read_text());checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    assert checkpoint['schema'] in ('full-racket-reach-cem-v4','full-racket-ball-cem-v4','full-racket-residual-ppo-v4') and checkpoint['parameters'].ndim==1
    if checkpoint['schema']=='full-racket-residual-ppo-v4':
        assert checkpoint['base_controller']['asset_sha256']==checkpoint['asset_sha256']
        assert torch.equal(checkpoint['parameters'],checkpoint['base_controller']['parameters'])
    assert checkpoint['asset_sha256']==protocol['asset']['sha256']
    assert digest(ROOT/protocol['asset']['path'])==protocol['asset']['sha256']
    dependencies=verify_dependency_manifest(ROOT)
    assert dependencies['asset']==protocol['asset']['path']
    for path,expected in protocol['physical_configs'].items():
        if digest(ROOT/path)!=expected:raise ValueError('Physical config changed: '+path)
    actor=json.loads((ROOT/'configs/g07_go2_walk_transfer.json').read_text())
    if digest(ROOT/actor['policy_path'])!=actor['policy_sha256']:raise ValueError('Source actor changed')
    files=list(ROOT.glob('quadsmash/**/*.py'))
    files += [a.protocol]+[ROOT/suite['path'] for suite in protocol['formal_suites'].values()]
    files += [ROOT/name for name in [*protocol['physical_configs'],protocol['asset']['path'],
        'configs/assembly_v2.json','configs/g07_go2_walk_transfer.json',actor['policy_path'],
        'scripts/evaluate_full_racket_v4.py','scripts/lock_full_racket_policy_v4.py','scripts/aggregate_full_racket_eval_v4.py']]
    files += list((ROOT/'assets/vendor/piper_isaac_sim/piper_description').rglob('*.STL'))
    files += list((ROOT/'assets/vendor/piper_isaac_sim/piper_description').rglob('*.urdf'))
    files += [ROOT/MANIFEST]
    files += [ROOT/record['path'] for record in dependencies['layers'] if not Path(record['path']).is_absolute()]
    files=sorted(set(files));hashes={str(path.relative_to(ROOT)):digest(path) for path in files}
    a.output.mkdir(parents=True)
    with zipfile.ZipFile(a.output/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as snapshot:
        for path in files:snapshot.write(path,str(path.relative_to(ROOT)))
        for record in dependencies['layers']:
            if Path(record['path']).is_absolute():
                snapshot.write(record['path'],record['archive_path'])
    with (a.output/'dirty.patch').open('w') as stream:subprocess.run(['git','diff','--binary'],cwd=ROOT,stdout=stream,check=True)
    lock=dict(schema='full-racket-policy-lock-v4',protocol_sha256=digest(a.protocol),
        policy_sha256=digest(a.checkpoint),asset_sha256=protocol['asset']['sha256'],source_hashes=hashes,
        git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_snapshot_sha256=digest(a.output/'source_snapshot.zip'),formal_panels=protocol['formal_suites'],
        asset_dependency_manifest_sha256=digest(ROOT/MANIFEST),asset_dependencies=dependencies['layers'],
        scope='Source and weight lock only; no policy performance claim.',formal_acceptance=False)
    (a.output/'LOCK.json').write_text(json.dumps(lock,indent=2)+'\n')
    print(a.output/'LOCK.json')


if __name__=='__main__':main()
