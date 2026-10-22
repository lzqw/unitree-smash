"""A real child process must save/stop and prevent subsequent chunk launch."""
import importlib.util
from pathlib import Path
import sys
import threading
import time

def test_driver_stop_forwards_and_prevents_next_job(tmp_path):
    spec=importlib.util.spec_from_file_location('g04_driver_test',Path(__file__).parents[1]/'scripts/continue_g04_fixed.py')
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    ready=tmp_path/'ready';saved=tmp_path/'saved'
    code="""import signal,sys,time
from pathlib import Path
def stop(*_):
    Path(sys.argv[2]).write_text('saved')
    raise SystemExit(0)
signal.signal(signal.SIGTERM,stop)
Path(sys.argv[1]).write_text('ready')
time.sleep(5)
"""
    def stop_after_ready():
        deadline=time.monotonic()+3
        while not ready.exists() and time.monotonic()<deadline:time.sleep(.01)
        driver.request_stop(None,None)
    thread=threading.Thread(target=stop_after_ready);thread.start()
    result=driver.run([sys.executable,'-c',code,str(ready),str(saved)])
    thread.join()
    assert result==130 and saved.read_text()=='saved' and driver.CHILD is None
    unintended=tmp_path/'must_not_launch'
    assert driver.run([sys.executable,'-c',"from pathlib import Path;import sys;Path(sys.argv[1]).touch()",str(unintended)])==130
    assert not unintended.exists()
