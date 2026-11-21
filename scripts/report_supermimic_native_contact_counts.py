"""Count new native contact edges from durable training chunks, independently of WM resampling."""
import argparse
import json
from pathlib import Path
import time

from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

from quadsmash.rl.supermimic_archive_storage_v1 import read_archive_rows
from quadsmash.rl.supermimic_archive_pipeline import native_event_counts


class NativeContactCounts:
    def __init__(self, arm):
        self.directory = arm/'train/native_transitions'
        self.report = arm/'native_contact_counts.json'
        self.chunks = json.loads(self.report.read_text())['chunks'] if self.report.exists() else []

    def update(self, writer):
        manifest = self.directory/'manifest.json'
        if not manifest.is_file():
            return
        data = json.loads(manifest.read_text())
        receipts = data['chunks']
        if len(self.chunks) > len(receipts) or any(
                c['sha256'] != r['sha256'] for c,r in zip(self.chunks,receipts)):
            raise ValueError('Published native chunk history changed')
        if len(receipts) == len(self.chunks):
            return
        totals = dict(qualified_contact=0, frame=0, qualified_or_frame=0)
        steps = 0
        for c in self.chunks:
            steps += c['env_steps']
            for key in totals:
                totals[key] += c[key]
        for receipt in receipts[len(self.chunks):]:
            entry = receipt.get('native_events')
            if entry is None:
                with read_archive_rows(self.directory, receipt, scratch=self.directory) as rows:
                    entry = native_event_counts(rows)
            counts = {k: entry[k] for k in totals}
            n = data['num_envs']*receipt['ticks']
            steps += n
            for key in totals:
                totals[key] += counts[key]
            self.chunks.append(dict(file=receipt['file'], sha256=receipt['sha256'], env_steps=n,
                                    **entry))
            writer.add_event(Event(wall_time=time.time(),step=steps,summary=Summary(value=[
                Summary.Value(tag='new_native_env_steps/native/'+k,simple_value=v) for k,v in totals.items()])))
        writer.flush()
        tmp = self.report.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(scope='Newly collected native lane-tick rising edges of qualified-contact and frame latches; repeated WM draws excluded.',
                                       durable_env_steps=steps, total=totals, chunks=self.chunks),indent=2)+'\n')
        tmp.replace(self.report)


if __name__ == '__main__':
    import psutil
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--arm-dir',required=True,type=Path)
    cli.add_argument('--tensorboard',required=True,type=Path)
    cli.add_argument('--follow-pid',type=int)
    args = cli.parse_args()
    owner = psutil.Process(args.follow_pid) if args.follow_pid else None
    counts = NativeContactCounts(args.arm_dir)
    writer = EventFileWriter(str(args.tensorboard))
    try:
        while True:
            live = owner and owner.is_running() and owner.status()!=psutil.STATUS_ZOMBIE
            counts.update(writer)
            if not live:
                break
            time.sleep(30)
    finally:
        writer.close()
