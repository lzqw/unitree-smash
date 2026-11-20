from types import SimpleNamespace
import hashlib
import pytest
import quadsmash.rl.supermimic_archive_storage_v1 as storage


def test_disk_guard_accounts_for_pending_chunk_before_creating_file(tmp_path,monkeypatch):
    monkeypatch.setattr(storage.shutil,'disk_usage',lambda _:SimpleNamespace(free=1<<20))
    with pytest.raises(OSError,match='pending_bound'):
        storage.write_archive_chunk(tmp_path/'chunk.pt',b'payload',reserve_bytes=0)
    assert list(tmp_path.iterdir())==[]


def test_atomic_raw_chunk_hash_and_no_overwrite(tmp_path):
    path=tmp_path/'chunk.pt';payload=b'deterministic serialized fixture'
    receipt=storage.write_archive_chunk(path,payload,reserve_bytes=0)
    assert path.read_bytes()==payload
    assert receipt['sha256']==hashlib.sha256(payload).hexdigest()
    assert not path.with_name(path.name+'.pending').exists()
    with pytest.raises(FileExistsError):storage.write_archive_chunk(path,b'replacement',reserve_bytes=0)
    assert path.read_bytes()==payload


def test_failed_compression_never_publishes_chunk_and_preserves_failure_artifact(tmp_path,monkeypatch):
    monkeypatch.setattr(storage.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=9))
    path=tmp_path/'chunk.pt.zst'
    with pytest.raises(RuntimeError,match='9'):storage.write_archive_chunk(path,b'payload',compress=True,reserve_bytes=0)
    assert not path.exists() and path.with_name(path.name+'.pending').exists()


@pytest.mark.parametrize('compressed',[False, True])
def test_resume_reads_verified_rows_and_releases_temporary_file(tmp_path, compressed):
    import io
    import torch
    data=io.BytesIO()
    torch.save(dict(rows=[{'sample':torch.arange(8)}]),data)
    name='chunk.pt.zst' if compressed else 'chunk.pt'
    receipt=storage.write_archive_chunk(tmp_path/name,data.getvalue(),compress=compressed,reserve_bytes=0)
    receipt.update(file=name,ticks=1)
    with storage.read_archive_rows(tmp_path,receipt,scratch=tmp_path) as rows:
        torch.testing.assert_close(rows[0]['sample'],torch.arange(8))
    assert list(tmp_path.iterdir())==[tmp_path/name]
    with pytest.raises(ValueError,match='hash mismatch'):
        with storage.read_archive_rows(tmp_path,dict(receipt,sha256='bad'),scratch=tmp_path):
            pass
