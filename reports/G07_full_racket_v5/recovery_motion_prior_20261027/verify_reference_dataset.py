"""Verify the separately published real6-case public-input diagnostic dataset."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import time
import urllib.request

HERE = Path(__file__).resolve().parent
MAIN = HERE.parents[2]


def main():
    expected = json.loads((HERE / 'REFERENCE_DATASET_CHECK.json').read_text())
    response = subprocess.run(['gh', 'api', 'repos/lzqw/unitree-smash/releases/422477709'],
        capture_output=True, text=True, check=True)
    assets = [row for row in json.loads(response.stdout)['assets'] if row['name'] == expected['asset']]
    assert len(assets) == 1
    asset = {key: assets[0][key] for key in ['id', 'name', 'size', 'state', 'digest', 'browser_download_url']}
    assert asset['state'] == 'uploaded' and asset['size'] == expected['bytes']
    assert asset['digest'] == 'sha256:' + expected['sha256']
    path = MAIN / 'reports/G07_full_racket_v4/contact_aim7_completed_delivery_20261027/download_verified.py'
    spec = importlib.util.spec_from_file_location('verified_release_transport', path)
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    transport.ASSET_ID = asset['id']
    for attempt in range(5):
        try:
            location = transport.signed_url()
            with urllib.request.urlopen(urllib.request.Request(location,
                    headers={'User-Agent': 'unitree-smash-reference-data-check'}), timeout=25) as stream:
                assert stream.status == 200
                data = stream.read(expected['bytes'] + 1)
            assert len(data) == expected['bytes'] and hashlib.sha256(data).hexdigest() == expected['sha256']
            break
        except Exception as error:
            print(json.dumps(dict(attempt=attempt + 1, category=type(error).__name__)), flush=True)
            if attempt == 4:
                raise RuntimeError('Reference-data download failed after five attempts') from None
            time.sleep(attempt + 1)
    downloaded = Path(tempfile.mkdtemp(prefix='g07_prior_public_reference_download_')) / expected['asset']
    downloaded.write_bytes(data)
    receipt = dict(checked_at=datetime.now(timezone.utc).isoformat(), asset=expected['asset'],
        GitHub_asset=asset, bytes=len(data), sha256=expected['sha256'], fresh_download=str(downloaded),
        fresh_download_sha256=hashlib.sha256(data).hexdigest(), upload_succeeded=True, download_verified=True,
        actual_public_rows=4800, original128_reference_outcomes_not_reestimated_from6_cases=True,
        program_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        new_native_performance=False, formal_acceptance=False)
    assert not (HERE / 'REFERENCE_DATASET_UPLOAD_RECEIPT.json').exists()
    transport.atomic(HERE / 'REFERENCE_DATASET_UPLOAD_RECEIPT.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(failed=True, category=type(error).__name__)), flush=True)
        raise SystemExit(1) from None
