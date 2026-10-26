"""Verify the authorized video upload by a fresh authenticated download."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request

HERE = Path(__file__).resolve().parent
VIDEO_REPORT = HERE.parents[3].parent / 'unitree_smash_video_framing/reports/G07_full_racket_v4/orthographic_leg06_warmup13_after_native_queue01'


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


def main():
    asset = json.loads((VIDEO_REPORT / 'GITHUB_UPLOAD_RESPONSE.json').read_text())
    review = json.loads((VIDEO_REPORT / 'ROOT_VISUAL_REVIEW.json').read_text())
    assert review['approved_for_authorized_GitHub_video_upload']
    assert asset['digest'] == 'sha256:' + review['sha256'] and asset['size'] == review['bytes']
    credential = subprocess.run(['gh', 'auth', 'token'], capture_output=True, text=True, check=True).stdout.strip()
    request = urllib.request.Request(
        f'https://api.github.com/repos/lzqw/unitree-smash/releases/assets/{asset["id"]}',
        headers={'Authorization': 'Bearer ' + credential, 'Accept': 'application/octet-stream',
                 'User-Agent': 'unitree-smash-video-verification'})
    try:
        response = urllib.request.build_opener(NoRedirect()).open(request, timeout=25)
    except urllib.error.HTTPError as error:
        if error.code not in (301, 302, 303, 307, 308):
            raise RuntimeError(f'GitHub redirect HTTP {error.code}') from None
        location = error.headers['Location']
    else:
        response.close()
        raise RuntimeError('Expected an artifact redirect')
    parsed = urllib.parse.urlparse(location)
    assert parsed.scheme == 'https' and parsed.hostname in {
        'release-assets.githubusercontent.com', 'objects.githubusercontent.com'}
    target = Path(tempfile.mkdtemp(prefix='g07_leg06_warmup13_video_fresh_20261026_')) / asset['name']
    with urllib.request.urlopen(location, timeout=40) as response, target.open('xb') as output:
        while block := response.read(1024 * 1024):
            output.write(block)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    assert digest == review['sha256'] and target.stat().st_size == asset['size']
    receipt = dict(checked_at=datetime.now(timezone.utc).isoformat(), repo='lzqw/unitree-smash',
        release='g07-full-racket-v4-progress-20261024', GitHub_asset_id=asset['id'],
        url=asset['browser_download_url'], bytes=asset['size'], sha256=digest,
        server_digest=asset['digest'], fresh_download=str(target), fresh_downloaded_sha256=digest,
        duration_s=review['duration_s'], frames=review['frames'], all4233_frames_decoded=True,
        root_reviewed_native_event_images=14, root_reviewed_decoded_frames=8,
        all_frames_visually_watched=False, upload_succeeded=True, download_verified=True,
        new_native_execution=False, formal_acceptance=False, goal_complete=False)
    path = HERE / 'UPLOAD_RECEIPT.json'
    assert not path.exists()
    path.with_suffix('.tmp').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    path.with_suffix('.tmp').replace(path)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(failed=True, category=type(error).__name__)), flush=True)
        raise SystemExit(1) from None
