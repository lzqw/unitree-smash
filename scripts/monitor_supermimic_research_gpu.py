"""Stream device and training-process resource measurements to TensorBoard."""
import argparse
import csv
import json
import shutil
import subprocess
import time
from pathlib import Path

import psutil
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter


def cleanup_finished_scratch(output, root, *, reserve=16*2**30):
    """Only inspect finished experiment directories under actual disk pressure."""
    free = shutil.disk_usage(root).free
    if free >= reserve:
        return free
    for marker in root.glob('supermimic*/**/training_completed.json'):
        for scratch in marker.parent.glob('replay_read_*'):
            if scratch.is_symlink() or not scratch.is_dir():
                continue
            # read_archive_rows creates just this disposable decompressed file.
            entries = list(scratch.iterdir())
            if any(p.name != 'chunk.pt' or p.is_symlink() or not p.is_file() for p in entries):
                continue
            if any(p.stat().st_mtime >= marker.stat().st_mtime for p in [scratch, *entries]):
                continue
            size = sum(p.stat().st_size for p in entries)
            for entry in entries:
                entry.unlink()
            scratch.rmdir()
            with (output/'disk_cleanup.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(path=str(scratch), bytes=size, reason='completed replay scratch'))+'\n')
            free = shutil.disk_usage(root).free
            if free >= reserve:
                return free
    return free


def monitor(pid, output, interval):
    process=psutil.Process(pid)
    process.cpu_percent()
    started=time.monotonic()
    writer=EventFileWriter(str(output),flush_secs=15)
    samples=0
    experiments = next((p for p in output.resolve().parents if p.name == 'experiments'), None)
    try:
        while process.is_running() and process.status()!=psutil.STATUS_ZOMBIE:
            raw=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=utilization.gpu,memory.used,memory.free,power.draw',
                                         '--format=csv,noheader,nounits'],text=True,timeout=10)
            util,used,free,power=map(float,next(csv.reader(raw.splitlines())))
            values={'gpu/utilization_percent':util,'gpu/used_MiB':used,'gpu/free_MiB':free,
                    'gpu/power_watts':power,'host/trainer_rss_GiB':process.memory_info().rss/2**30,
                    'host/trainer_cpu_cores':process.cpu_percent()/100,'host/available_GiB':psutil.virtual_memory().available/2**30}
            if experiments is not None and samples % 12 == 0:
                values['host/disk_available_GiB'] = cleanup_finished_scratch(output, experiments)/2**30
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
