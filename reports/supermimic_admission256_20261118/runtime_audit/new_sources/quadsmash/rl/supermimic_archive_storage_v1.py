"""Bounded archive writes; preserve pending bytes and never publish partial chunks."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess


RESERVE_BYTES=8*1024**3


def require_archive_space(directory,pending_bytes=0,*,reserve_bytes=RESERVE_BYTES):
    if pending_bytes<0 or reserve_bytes<0:raise ValueError('Negative archive space estimate')
    free=shutil.disk_usage(directory).free
    # Covers serialization/compressor overhead, without assuming a favorable
    # compression ratio. The reserve remains available for stop/checkpoint logs.
    allowance=int(pending_bytes)+int(pending_bytes)//100+(1<<20 if pending_bytes else 0)
    if free<reserve_bytes+allowance:
        raise OSError(f'Archive disk guard: free={free}, pending_bound={allowance}, reserve={reserve_bytes}')
    return dict(free_before_bytes=free,pending_bound_bytes=allowance,reserve_bytes=reserve_bytes)


def write_archive_chunk(path,payload,*,compress=False,reserve_bytes=RESERVE_BYTES):
    path=Path(path);temporary=path.with_name(path.name+'.pending')
    if path.exists() or temporary.exists():raise FileExistsError(path if path.exists() else temporary)
    budget=require_archive_space(path.parent,len(payload),reserve_bytes=reserve_bytes)
    with temporary.open('xb') as stream:
        if compress:
            # Two CPU workers, bounded independently of host core count.
            result=subprocess.run(['zstd','-q','-T2','-3','-c'],input=payload,stdout=stream,check=False)
            if result.returncode:raise RuntimeError(f'zstd archive compression failed: {result.returncode}')
        else:stream.write(payload)
        stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):digest.update(block)
    return dict(sha256=digest.hexdigest(),serialized_bytes=len(payload),bytes=path.stat().st_size,storage_guard=budget)
