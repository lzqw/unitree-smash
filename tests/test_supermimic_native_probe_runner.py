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
    assert env['PYTHONPATH'].split(':')[:1] == ['/home/autodrive/supermimic_work_20261118']
    assert env['PYTHONPATH'].endswith(':/existing')
    assert env['PYTHONUNBUFFERED'] == '1'
