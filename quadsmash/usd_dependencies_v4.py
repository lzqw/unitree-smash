"""Hash every composed robot layer, including USD files outside the repo."""
import hashlib
import json
from pathlib import Path

MANIFEST = 'configs/full_racket_usd_dependencies_v4.json'


def verify_dependency_manifest(root, manifest_path=None):
    root = Path(root)
    path = Path(manifest_path) if manifest_path is not None else root/MANIFEST
    manifest = json.loads(path.read_text())
    if manifest['schema'] != 'full-racket-usd-dependencies-v4':
        raise ValueError('Unknown USD dependency manifest')
    records = manifest['layers']
    if not records or len({x['path'] for x in records}) != len(records):
        raise ValueError('Empty or duplicate USD dependency list')
    for record in records:
        dependency = root/record['path']
        if (not dependency.is_file() or dependency.stat().st_size != record['bytes'] or
                hashlib.sha256(dependency.read_bytes()).hexdigest() != record['sha256']):
            raise ValueError('USD dependency missing or changed: '+record['path'])
    return manifest


def audit_composed_asset(root, asset):
    """Use the running USD library to compare composition with the pinned set."""
    from pxr import Usd
    root = Path(root)
    manifest = verify_dependency_manifest(root)
    asset = Path(asset).resolve()
    if asset != (root/manifest['asset']).resolve():
        raise ValueError('Unexpected robot asset in dependency audit')
    stage = Usd.Stage.Open(str(asset))
    if stage is None:
        raise ValueError('Cannot open composed robot asset')
    actual = {str(Path(layer.realPath).resolve()) for layer in stage.GetUsedLayers() if layer.realPath}
    expected = {str((root/record['path']).resolve()) for record in manifest['layers']}
    if actual != expected:
        raise ValueError('USD composition changed: missing='+str(sorted(expected-actual))+
                         ', extra='+str(sorted(actual-expected)))
    return dict(scope='Composed USD source-layer integrity only, not policy performance',
                asset=str(asset), layer_count=len(actual), all_layers_pinned=True,
                resolved_layers=sorted(actual), manifest_sha256=hashlib.sha256((root/MANIFEST).read_bytes()).hexdigest(),
                visual_shader_caveat=manifest['visual_shader_caveat'], formal_acceptance=False)
