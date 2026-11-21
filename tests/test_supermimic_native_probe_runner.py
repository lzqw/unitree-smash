from pathlib import Path
import pytest

from scripts.run_supermimic_event_aligned_native_probes import child_environment, kit_args_option


def test_kit_args_starting_with_dash_remain_one_argument():
    assert kit_args_option('--renderer/multiGpu/enabled=false') == (
        '--kit_args=--renderer/multiGpu/enabled=false'
    )


def test_empty_kit_args_are_rejected():
    with pytest.raises(ValueError, match='must not be empty'):
        kit_args_option('')


def test_child_environment_explicitly_exports_project_import_path():
    env = child_environment({'PYTHONPATH': '/existing', 'PATH': '/bin'})
    assert env['PYTHONPATH'].split(':')[:1] == [str(Path(__file__).resolve().parents[1])]
    assert env['PYTHONPATH'].endswith(':/existing')
    assert env['PYTHONUNBUFFERED'] == '1'


def test_resume_rejects_wrong_spec_or_corrupt_data(tmp_path):
    from scripts.run_supermimic_event_aligned_native_probes import completed_probe_hash, sha256
    import json

    probe = tmp_path / 'native_probe.pt'
    spec = {'tick': 271}
    probe.write_bytes(b'probe')
    digest = sha256(probe)
    (tmp_path / 'native_probe_manifest.json').write_text(json.dumps(
        {'sha256': digest, 'spec': spec, 'branch': 'nominal_a'}))
    assert completed_probe_hash(tmp_path, 'nominal_a', spec) is None
    (tmp_path / 'shutdown_diagnostic.json').write_text('{"control_ticks": 272}')
    assert completed_probe_hash(tmp_path, 'nominal_a', spec) == digest
    assert completed_probe_hash(tmp_path, 'nominal_b', spec) is None
    probe.write_bytes(b'corrupt')
    assert completed_probe_hash(tmp_path, 'nominal_a', spec) is None


def test_resume_detects_recorded_source_drift_even_with_an_intact_archive(tmp_path):
    import torch
    from scripts.run_supermimic_event_aligned_native_probes import captured_source_mismatch, source_mismatch
    expected = {'ball.py': 'new', 'executor.py': 'stable'}
    path = tmp_path/'native_probe.pt'
    torch.save(dict(execution_code_sha256=dict(expected)), path)
    assert captured_source_mismatch(path, expected) is None
    assert 'ball.py' in captured_source_mismatch(path, dict(expected, **{'ball.py':'old'}))
    assert source_mismatch(expected, None)
