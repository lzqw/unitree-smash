"""Compress one immutable native chunk while the next control ticks run."""
from concurrent.futures import ThreadPoolExecutor
import io
import time

import torch

from .supermimic_archive_storage_v1 import require_archive_space, write_archive_chunk
from .supermimic_physical_capture_v1 import tensor_tree_copy


def serialize(directory, name, schema, snapshot):
    started = time.perf_counter()
    payload = io.BytesIO()
    torch.save(dict(schema=schema, rows=snapshot), payload)
    receipt = write_archive_chunk(directory / name, payload.getbuffer(), compress=True)
    return dict(file=name, ticks=len(snapshot), compression='zstd', **receipt), time.perf_counter() - started


class ArchivePipeline:
    def __init__(self, archive):
        self.archive = archive
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='native-replay-write')
        self.pending = None
        self.snapshot_seconds = self.worker_seconds = self.wait_seconds = 0.

    def timing(self):
        return dict(snapshot_seconds=self.snapshot_seconds, worker_seconds=self.worker_seconds,
                    wait_seconds=self.wait_seconds, chunks=len(self.archive.chunks))

    def drain(self):
        if self.pending is not None:
            started = time.perf_counter()
            # Keep a failed future so every later checkpoint attempt also fails.
            receipt, elapsed = self.pending.result()
            self.wait_seconds += time.perf_counter() - started
            self.worker_seconds += elapsed
            self.archive.chunks.append(receipt)
            self.pending = None

    def enqueue(self):
        self.drain()
        archive = self.archive
        if not archive.rows:
            return
        require_archive_space(archive.directory)
        started = time.perf_counter()
        snapshot = tensor_tree_copy(archive.rows, cpu=True)
        self.snapshot_seconds += time.perf_counter() - started
        name = f'chunk_{len(archive.chunks):05d}.pt.zst'
        self.pending = self.worker.submit(serialize, archive.directory, name, archive.schema, snapshot)
        archive.rows.clear()

    def close(self):
        try:
            self.drain()
        finally:
            self.worker.shutdown(wait=True)
