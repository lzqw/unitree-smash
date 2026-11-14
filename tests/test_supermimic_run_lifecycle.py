import json
import pytest
from quadsmash.rl.supermimic_run_lifecycle import close_native, inspect_process_result


class Resource:
    def __init__(self,error=False):
        self.error=error
        self.closed=0
    def close(self):
        self.closed+=1
        if self.error:
            raise RuntimeError('cleanup fixture error')


def test_cleanup_failure_attempts_app_and_is_not_swallowed(tmp_path):
    env,app=Resource(True),Resource()
    with pytest.raises(RuntimeError,match='Native cleanup failed'):
        close_native(tmp_path,env,app,primary_failed=False)
    assert env.closed==app.closed==1
    assert 'cleanup fixture error' in (tmp_path/'env_cleanup_failure.txt').read_text()
    assert not inspect_process_result(tmp_path,0)['process_completed_cleanly']


def test_cleanup_does_not_replace_primary_exception(tmp_path):
    env,app=Resource(True),Resource(True)
    close_native(tmp_path,env,app,primary_failed=True)
    assert env.closed==app.closed==1
    assert (tmp_path/'app_cleanup_failure.txt').exists()


@pytest.mark.parametrize('returncode,expected', [(0,0),(-11,139),(13,13)])
def test_completed_file_cannot_hide_process_crash(tmp_path,returncode,expected):
    (tmp_path/'completed.json').write_text(json.dumps(dict(control_ticks=800,episodes=128)))
    close_native(tmp_path,Resource(),Resource(),primary_failed=False)
    result=inspect_process_result(tmp_path,returncode)
    assert result['diagnostic_exit_code']==expected
    assert result['process_completed_cleanly']==(returncode==0)


def test_zero_exit_without_complete_evidence_fails(tmp_path):
    assert inspect_process_result(tmp_path,0)['diagnostic_exit_code']==1
    (tmp_path/'completed.json').write_text(json.dumps(dict(control_ticks=800,episodes=128)))
    assert inspect_process_result(tmp_path,0)['diagnostic_exit_code']==1
    close_native(tmp_path,None,Resource(),primary_failed=False)
    (tmp_path/'failure.json').write_text('{}')
    assert not inspect_process_result(tmp_path,0)['process_completed_cleanly']


@pytest.mark.parametrize('fail', [False, True])
def test_scoped_native_releases_cycles_and_exception_locals_before_app(tmp_path, fail):
    import weakref
    from quadsmash.rl.supermimic_run_lifecycle import execute_scoped_native
    refs = []
    class NativeObject:
        pass
    class App:
        def close(self):
            assert refs and all(ref() is None for ref in refs)
    def run():
        native = NativeObject()
        native.cycle = native
        refs.append(weakref.ref(native))
        if fail:
            raise ValueError('original simulation failure')
    if fail:
        with pytest.raises(ValueError, match='original simulation failure'):
            execute_scoped_native(tmp_path, run, App())
        assert 'original simulation failure' in (tmp_path/'scoped_failure_traceback.txt').read_text()
    else:
        execute_scoped_native(tmp_path, run, App())
    stages = [json.loads(line)['stage'] for line in (tmp_path/'lifecycle.jsonl').read_text().splitlines()]
    assert stages.index('AFTER_NATIVE_GC') < stages.index('BEFORE_APP_CLOSE')


def test_shutdown_diagnostic_cannot_be_reported_as_formal_evaluation(tmp_path):
    (tmp_path/'shutdown_diagnostic.json').write_text(json.dumps(dict(
        control_ticks=10, requested_ticks=10, num_envs=128)))
    close_native(tmp_path, None, Resource(), primary_failed=False)
    assert not inspect_process_result(tmp_path, 0)['process_completed_cleanly']
    result = inspect_process_result(tmp_path, 0, shutdown_diagnostic_ticks=10)
    assert result['process_completed_cleanly']
    assert not result['simulation_completed']
    assert result['shutdown_diagnostic_completed']
    assert not inspect_process_result(tmp_path, -11, shutdown_diagnostic_ticks=10)['process_completed_cleanly']
    assert not inspect_process_result(tmp_path, 0, shutdown_diagnostic_ticks=11)['process_completed_cleanly']
    (tmp_path/'metrics.json').write_text('{}')
    assert not inspect_process_result(tmp_path, 0, shutdown_diagnostic_ticks=10)['process_completed_cleanly']


def test_kit_error_cache_is_reset_before_collect_and_app_close(tmp_path, monkeypatch):
    import sys
    import types
    import weakref
    from quadsmash.rl.supermimic_run_lifecycle import execute_scoped_native
    cache = {}
    refs = []
    class NativeResource:
        pass
    def run():
        native = NativeResource()
        refs.append(weakref.ref(native))
        try:
            raise FileNotFoundError('negative import lookup')
        except FileNotFoundError as error:
            cache['missing'] = (None, error)
    reset_calls = []
    def reset():
        reset_calls.append(True)
        cache.clear()
    module = types.SimpleNamespace(_stat_cache=cache, reset_stat_cache=reset)
    monkeypatch.setitem(sys.modules, 'omni.ext._impl.stat_cache', module)
    class App:
        def close(self):
            assert reset_calls == [True]
            assert refs[0]() is None
    execute_scoped_native(tmp_path, run, App())
    events = [json.loads(line) for line in (tmp_path/'lifecycle.jsonl').read_text().splitlines()]
    assert next(e for e in events if e['stage']=='BEFORE_KIT_STAT_CACHE_RESET')['exceptions'] == 1
    assert next(e for e in events if e['stage']=='AFTER_KIT_STAT_CACHE_RESET')['entries'] == 0
