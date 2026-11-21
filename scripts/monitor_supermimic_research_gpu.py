"""Stream device and training-process resource measurements to TensorBoard."""
import argparse
import csv
import subprocess
import time
from pathlib import Path

import psutil
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter


def monitor(pid, output, interval):
    process=psutil.Process(pid)
    process.cpu_percent()
    started=time.monotonic()
    writer=EventFileWriter(str(output),flush_secs=15)
    samples=0
    try:
        while process.is_running() and process.status()!=psutil.STATUS_ZOMBIE:
            raw=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=utilization.gpu,memory.used,memory.free,power.draw',
                                         '--format=csv,noheader,nounits'],text=True,timeout=10)
            util,used,free,power=map(float,next(csv.reader(raw.splitlines())))
            values={'gpu/utilization_percent':util,'gpu/used_MiB':used,'gpu/free_MiB':free,
                    'gpu/power_watts':power,'host/trainer_rss_GiB':process.memory_info().rss/2**30,
                    'host/trainer_cpu_cores':process.cpu_percent()/100,'host/available_GiB':psutil.virtual_memory().available/2**30}
            writer.add_event(Event(wall_time=time.time(),step=int(time.monotonic()-started),
                summary=Summary(value=[Summary.Value(tag=k,simple_value=v) for k,v in values.items()])))
            samples+=1
            if samples==1:writer.flush()
            time.sleep(interval)
    except psutil.NoSuchProcess:
        pass
    finally:
        writer.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pid',type=int,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--interval',type=float,default=5.)
    args=ap.parse_args()
    if args.interval<=0:ap.error('Positive sampling interval required')
    monitor(args.pid,args.output,args.interval)
