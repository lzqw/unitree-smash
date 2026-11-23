import hashlib
import json

import pytest

from quadsmash.rl.supermimic_resident_gate import await_native_decision, publish_boundary


def test_resident_boundary_is_immutable_and_bound_to_decision(tmp_path):
    latest = tmp_path / 'latest.pt'
    latest.write_bytes(b'original checkpoint')
    directory = tmp_path / 'gate'
    receipt = publish_boundary(directory, latest, 40, {'free_bytes': 1024})
    next_checkpoint = tmp_path / 'next.pt'
    next_checkpoint.write_bytes(b'replacement checkpoint')
    next_checkpoint.replace(latest)
    assert (directory / 'policy.pt').read_bytes() == b'original checkpoint'
    assert json.loads((directory / 'ready.json').read_text()) == receipt
    assert await_native_decision(directory, receipt, 0, lambda: None) == 'stop'

    paired = b'{"result": "training-panel-pair"}'
    (directory / 'paired.json').write_bytes(paired)
    decision = dict(update=40, checkpoint_sha256=receipt['checkpoint_sha256'],
                    action='continue', paired_result_sha256=hashlib.sha256(paired).hexdigest())
    (directory / 'decision.json').write_text(json.dumps(decision))
    assert await_native_decision(directory, receipt, 1, lambda: None) == 'continue'
    (directory / 'decision.json').write_text(json.dumps(dict(decision, update=48)))
    with pytest.raises(ValueError, match='does not match'):
        await_native_decision(directory, receipt, 1, lambda: None)
