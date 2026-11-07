"""Losslessly archive a completed local diagnostic log, retaining restoration metadata."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil


def digest(path, compressed=False):
    h=hashlib.sha256()
    with (gzip.open(path,'rb') if compressed else open(path,'rb')) as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def archive(path):
    path=Path(path)
    if path.name!='first_nonfoot_contact_details.json' or path.is_symlink():
        raise ValueError('Only explicit local generated contact logs may be archived')
    job=json.loads((path.parent/'job.json').read_text())
    if job.get('status') not in ('SUCCEEDED','FAILED'):
        raise ValueError('Writer must be terminal')
    if any(Path('/proc',str(job.get(k,-1))).exists() for k in ('pid','child_pid')):
        raise ValueError('Writer process is still present')
    packed=path.with_suffix('.json.gz');receipt=path.with_suffix('.archive.json')
    if packed.exists() or receipt.exists():raise ValueError('Preserve existing archive')
    original_hash=digest(path);before=path.stat()
    with open(path,'rb') as source,open(packed,'xb') as out:
        with gzip.GzipFile(filename='',fileobj=out,mode='wb',compresslevel=3,mtime=0) as zipped:
            shutil.copyfileobj(source,zipped,1024*1024)
        out.flush();os.fsync(out.fileno())
    if digest(packed,compressed=True)!=original_hash or digest(path)!=original_hash:
        raise ValueError('Archive round trip or source identity failed; original retained')
    record=dict(original_name=path.name,archive_name=packed.name,original_sha256=original_hash,
        archive_sha256=digest(packed),original_bytes=before.st_size,archive_bytes=packed.stat().st_size,
        original_mode=before.st_mode&0o777,original_mtime_ns=before.st_mtime_ns,
        lossless_round_trip_verified=True,writer_job_sha256=digest(path.parent/'job.json'))
    with receipt.open('x') as f:
        json.dump(record,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    path.unlink()
    return record


def restore(receipt):
    receipt=Path(receipt);record=json.loads(receipt.read_text())
    path=receipt.parent/record['original_name'];packed=receipt.parent/record['archive_name']
    if path.exists() or digest(packed)!=record['archive_sha256']:raise ValueError('Destination exists or archive changed')
    temporary=path.with_suffix('.restore.tmp')
    with gzip.open(packed,'rb') as source,temporary.open('xb') as out:
        shutil.copyfileobj(source,out,1024*1024);out.flush();os.fsync(out.fileno())
    if digest(temporary)!=record['original_sha256']:raise ValueError('Restored content mismatch')
    os.chmod(temporary,record['original_mode']);os.utime(temporary,ns=(record['original_mtime_ns'],record['original_mtime_ns']))
    temporary.replace(path)
    return path


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=['archive','restore']);p.add_argument('path',type=Path)
    a=p.parse_args();print(json.dumps(archive(a.path)) if a.operation=='archive' else str(restore(a.path)))
