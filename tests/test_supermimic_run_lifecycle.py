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
