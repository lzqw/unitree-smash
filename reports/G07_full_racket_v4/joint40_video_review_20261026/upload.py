"""Upload visually reviewed real failure videos and verify fresh downloads."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
REPO='lzqw/unitree-smash';TAG='g07-full-racket-v4-progress-20261024'


def main():
    path=HERE/'VIDEO_UPLOAD_RECEIPT.json';assert not path.exists()
    review=json.loads((HERE/'VISUAL_REVIEW.json').read_text());assert review['status']=='APPROVED_FOR_UPLOAD'
    outputs=[]
    for video in review['videos']:
        source=ROOT/video['video'];assert hashlib.sha256(source.read_bytes()).hexdigest()==video['sha256']
        command=['gh','api',f'repos/{REPO}/releases/tags/{TAG}']
        release=json.loads(subprocess.check_output(command,text=True))
        matches=[asset for asset in release['assets']if asset['name']==source.name]
        if not matches:
            subprocess.run(['gh','release','upload',TAG,str(source),'--repo',REPO],check=True)
            release=json.loads(subprocess.check_output(command,text=True))
            matches=[asset for asset in release['assets']if asset['name']==source.name]
        assert len(matches)==1 and matches[0]['size']==source.stat().st_size
        directory=Path(tempfile.mkdtemp(prefix='g07_joint40_video_verify_'))
        subprocess.run(['gh','release','download',TAG,'--repo',REPO,'--pattern',source.name,'--dir',str(directory)],check=True)
        downloaded=directory/source.name;digest=hashlib.sha256(downloaded.read_bytes()).hexdigest()
        assert digest==video['sha256'] and downloaded.stat().st_size==video['bytes']
        result=dict(url=matches[0]['browser_download_url'],release_asset_id=matches[0]['id'],
            video=video['video'],bytes=video['bytes'],sha256=video['sha256'],downloaded_sha256=digest,
            download_verified=True,duration_s=video['duration_s'],frames=video['frames_decoded'])
        outputs.append(result);print(json.dumps(result),flush=True)
    value=dict(verified_at=datetime.now(timezone.utc).isoformat(),repo=REPO,tag=TAG,videos=outputs,
        actual_fixed_TRAIN128_complete=0,actual_fixed_TRAIN128_airborne_contacts=0,
        scope='Real failed frozen40 executions; TRAIN65/6 are retests, not reconstructed successful training motion.',
        formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    path.write_text(json.dumps(value,indent=2)+'\n')


if __name__=='__main__':main()
