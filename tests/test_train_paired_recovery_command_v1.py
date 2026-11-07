import json
import hashlib
import numpy as np
import pytest
import torch
from scripts.train_paired_recovery_command_v1 import sample_batch, validate_provenance


def episode(marker, corrected):
    return dict(features=torch.full((103, 831), float(marker)),
                delta=torch.full((103, 18), float(marker) / 10.), corrected=corrected)


def test_sampling_keeps_retention_weight_when_new_dataset_is_larger():
    old = [episode(1, False), episode(2, True)]
    new = [episode(3, True) for _ in range(100)]
    features, target = sample_batch([old, new], np.random.default_rng(1), 512)
    assert int((features[:, 0] < 3).sum()) == 256
    assert int((features[:, 0] == 3).sum()) == 256
    torch.testing.assert_close(target[:, 0], features[:, 0] / 10.)
    with pytest.raises(ValueError):
        sample_batch([old, []], np.random.default_rng(1), 512)


def test_provenance_requires_actual_changed_teacher_commands(tmp_path):
    manifest = dict(failed_episodes_admitted=0, teacher_increment_is_total_label=False,
                    teacher_changed_rows=0, source_sha256={})
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    validation = dict(passed=True, trainable=True,
                      manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (tmp_path / 'validation.json').write_text(json.dumps(validation))
    with pytest.raises(ValueError, match='actual teacher corrections'):
        validate_provenance(tmp_path, manifest)


def test_provenance_rejects_changed_execution_evidence(tmp_path):
    source = tmp_path / 'source.json'
    source.write_text('original')
    manifest = dict(failed_episodes_admitted=0, teacher_increment_is_total_label=False,
        teacher_changed_rows=1, source_sha256={str(source): hashlib.sha256(source.read_bytes()).hexdigest()})
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    (tmp_path / 'validation.json').write_text(json.dumps(dict(passed=True, trainable=True,
        manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
    validate_provenance(tmp_path, manifest)
    source.write_text('modified')
    with pytest.raises(ValueError, match='evidence changed'):
        validate_provenance(tmp_path, manifest)
