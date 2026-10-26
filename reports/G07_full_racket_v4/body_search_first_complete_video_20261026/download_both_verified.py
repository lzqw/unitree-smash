"""Fresh-download both authorized videos without exposing signed URLs."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[3]
FOLDERS=[ROOT/'reports/G07_full_racket_v4/body_search_first_complete_video_20261026',
    ROOT/'reports/G07_full_racket_v4/orthographic_fixed_actor_body_gen2_after_native_queue01']


def main():
    for folder in FOLDERS:
        receipt=folder/'UPLOAD_RECEIPT.json'
        if receipt.exists():
            assert json.loads(receipt.read_text())['download_verified'];continue
        asset=json.loads((folder/'GITHUB_UPLOAD_RESPONSE.json').read_text())
        review=json.loads((folder/'ROOT_VISUAL_REVIEW.json').read_text())
        assert review['approved_for_authorized_GitHub_video_upload']
        assert asset['digest']=='sha256:'+review['sha256'] and asset['size']==review['bytes']
        target=Path(tempfile.mkdtemp(prefix='g07_body_video_fresh_20261026_'))/asset['name']
        with target.open('xb') as out:
            result=subprocess.run(['gh','api',f'repos/lzqw/unitree-smash/releases/assets/{asset["id"]}',
                '--header','Accept: application/octet-stream'],stdout=out,stderr=subprocess.PIPE,
                env=dict(os.environ,GODEBUG='http2client=0'),timeout=120)
        assert result.returncode==0,'Authenticated asset download failed; preserve partial file'
        digest=hashlib.sha256(target.read_bytes()).hexdigest()
        assert target.stat().st_size==asset['size'] and digest==review['sha256']
        data=dict(checked_at=datetime.now(timezone.utc).isoformat(),repo='lzqw/unitree-smash',
            release='g07-full-racket-v4-progress-20261024',GitHub_asset_id=asset['id'],
            url=asset['browser_download_url'],bytes=asset['size'],sha256=digest,
            server_digest=asset['digest'],fresh_download=str(target),fresh_downloaded_sha256=digest,
            duration_s=review['duration_s'],frames=review['frames'],all_frames_decoded=True,
            root_reviewed_native_event_images=review['root_reviewed_native_event_images'],
            root_reviewed_decoded_frames=8,all_frames_visually_watched=False,
            upload_succeeded=True,download_verified=True,formal_acceptance=False,goal_complete=False)
        receipt.write_text(json.dumps(data,indent=2)+'\n')
        print(json.dumps(data),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        print(json.dumps(dict(failed=True,category=type(error).__name__)),flush=True)
        raise SystemExit(1) from None
