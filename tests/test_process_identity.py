import json
import subprocess
import sys
import tempfile
from pathlib import Path
import pytest
from quadsmash.process_identity import write_identity,checked_pid

def test_live_child_identity_and_recycled_pid_guard():
    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(10)'])
    try:
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);(directory/'run.pid').write_text(str(child.pid))
            write_identity(directory,child.pid)
            assert checked_pid(directory)==child.pid
            path=directory/'process.json';record=json.loads(path.read_text())
            record['process']['start_ticks']+=1;path.write_text(json.dumps(record))
            with pytest.raises(RuntimeError,match='stale'):
                checked_pid(directory)
    finally:
        child.terminate();child.wait()
