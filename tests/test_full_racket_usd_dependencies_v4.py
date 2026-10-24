"""External-layer mutation must prevent evaluation before any rollout."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from quadsmash.usd_dependencies_v4 import verify_dependency_manifest


class ExternalLayerIntegrityTest(unittest.TestCase):
    def test_external_layer_change_or_deletion_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);external=root/'external.usd';external.write_bytes(b'original robot')
            manifest=root/'manifest.json'
            manifest.write_text(json.dumps(dict(schema='full-racket-usd-dependencies-v4',layers=[dict(
                path=str(external),bytes=external.stat().st_size,
                sha256=hashlib.sha256(external.read_bytes()).hexdigest())])))
            verify_dependency_manifest(root,manifest)
            external.write_bytes(b'modified robot')
            with self.assertRaisesRegex(ValueError,'missing or changed'):
                verify_dependency_manifest(root,manifest)
            external.unlink()
            with self.assertRaisesRegex(ValueError,'missing or changed'):
                verify_dependency_manifest(root,manifest)
