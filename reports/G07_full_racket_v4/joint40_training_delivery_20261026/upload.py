"""Upload the immutable evidence bundle and verify a fresh authenticated download."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

HERE=Path(__file__).resolve().parent
REPO='lzqw/unitree-smash';TAG='g07-full-racket-v4-progress-20261024'


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb')as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def main():
    receipt=HERE/'UPLOAD_RECEIPT.json';assert not receipt.exists()
    metadata=json.loads((HERE/'ARCHIVE.json').read_text());path=Path(metadata['archive'])
    assert sha(path)==metadata['sha256'] and path.stat().st_size==metadata['bytes']
    command=['gh','api',f'repos/{REPO}/releases/tags/{TAG}']
    remote=json.loads(subprocess.check_output(command,text=True))
    found=[asset for asset in remote['assets']if asset['name']==path.name]
    if not found:
        subprocess.run(['gh','release','upload',TAG,str(path),'--repo',REPO],check=True)
        remote=json.loads(subprocess.check_output(command,text=True))
        found=[asset for asset in remote['assets']if asset['name']==path.name]
    assert len(found)==1 and found[0]['size']==metadata['bytes']
    directory=Path(tempfile.mkdtemp(prefix='g07_joint40_delivery_verify_'))
    subprocess.run(['gh','release','download',TAG,'--repo',REPO,'--pattern',path.name,'--dir',str(directory)],check=True)
    downloaded=directory/path.name;assert downloaded.stat().st_size==metadata['bytes']
    digest=sha(downloaded);assert digest==metadata['sha256']
    result=dict(verified_at=datetime.now(timezone.utc).isoformat(),repo=REPO,tag=TAG,
        url=found[0]['browser_download_url'],release_asset_id=found[0]['id'],
        bytes=metadata['bytes'],sha256=metadata['sha256'],downloaded_sha256=digest,download_verified=True,
        independent_download=str(downloaded),members=metadata['members'],
        scope='Completed changing-policy training evidence only. Frozen40 assessment and new videos were not included while live.',
        formal_acceptance=False,goal_complete=False)
    receipt.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
