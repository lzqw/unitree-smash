import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts.archive_completed_contact_log_v1 import archive,restore,digest
from scripts.native_output_reserve_v1 import NativeOutputReserve


def test_archive_restores_exact_bytes_and_rejects_corruption(tmp_path):
    (tmp_path/'job.json').write_text(json.dumps(dict(status='SUCCEEDED')))
    path=tmp_path/'first_nonfoot_contact_details.json';data=b'{"observed": true}\n'*1000;path.write_bytes(data)
    record=archive(path)
    assert not path.exists() and record['lossless_round_trip_verified']
    receipt=tmp_path/'first_nonfoot_contact_details.archive.json'
    restored=restore(receipt);assert restored.read_bytes()==data
    restored.unlink();packed=tmp_path/record['archive_name']
    with packed.open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError):restore(receipt)
    assert not path.exists()


def test_active_writer_is_never_archived(tmp_path):
    import os
    (tmp_path/'job.json').write_text(json.dumps(dict(status='SUCCEEDED',pid=os.getpid())))
    path=tmp_path/'first_nonfoot_contact_details.json';path.write_text('{}')
    with pytest.raises(ValueError):archive(path)
    assert path.read_text()=='{}'


def test_pressure_releases_reserved_blocks_and_fails_explicitly(tmp_path,monkeypatch):
    reserve=NativeOutputReserve(tmp_path,reserve_bytes=1024**2,minimum_free_bytes=4*1024**2,pressure_bytes=1024**2)
    assert reserve.path.stat().st_blocks*512>=1024**2
    monkeypatch.setattr('scripts.native_output_reserve_v1.shutil.disk_usage',lambda p:SimpleNamespace(free=0))
    with pytest.raises(OSError,match='Storage pressure'):reserve.check()
    assert not reserve.path.exists();reserve.release()


def test_no_space_refuses_job_before_allocating(tmp_path,monkeypatch):
    monkeypatch.setattr('scripts.native_output_reserve_v1.shutil.disk_usage',lambda p:SimpleNamespace(free=0))
    with pytest.raises(OSError,match='Insufficient'):NativeOutputReserve(tmp_path)
    assert not (tmp_path/'.native_final_output.reserve').exists()
