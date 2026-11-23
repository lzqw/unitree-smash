"""Restore Approach Replay from derived tensors keyed by immutable archive chunks."""

import hashlib
import inspect
import json
from pathlib import Path
import pickle
import tempfile

import torch

from .supermimic_approach_data_v1 import FIELDS
from .supermimic_archive_storage_v1 import read_archive_rows, require_archive_space, verify_archive_chunk
from .supermimic_physical_loop_replay_v1 import SOURCE_CODES, _stage_ids_from_tensors, row_tensors


def packing_identity(schema):
    from .supermimic_physical_dataset_v1 import encode_physical

    digest = hashlib.sha256(json.dumps(schema.manifest(), sort_keys=True).encode())
    for function in (encode_physical, schema.encode, row_tensors, pack_archive_rows):
        digest.update(inspect.getsource(function).encode())
    return digest.hexdigest()


def pack_archive_rows(rows, schema):
    records = [row_tensors(row, schema) for row in rows]
    if not records:
        raise ValueError('Empty Replay archive chunk')
    lanes = len(records[0]['state'])
    fields = FIELDS + tuple('next_' + name for name in FIELDS[2:])
    absence = torch.full((lanes,), torch.nan)
    not_perturbed = torch.zeros(lanes, dtype=torch.bool)
    return dict(data={name: torch.stack([record[name] for record in records]) for name in records[0]},
                versions=torch.tensor([row.get('policy_version', 0) for row in rows], dtype=torch.long),
                approach={name: torch.stack([row.get('sampling_metadata', {}).get(name, absence)
                                             for row in rows]).float() for name in fields},
                excitation=torch.stack([row.get('arm_identification_active', not_perturbed).reshape(lanes)
                                        for row in rows]).bool())


def read_packed_chunk(directory, receipt, *, schema, cache, identity, scratch):
    if cache is None:
        with read_archive_rows(directory, receipt, scratch=scratch) as rows:
            return pack_archive_rows(rows, schema), False
    path = Path(cache) / identity / (receipt['sha256'] + '.pt')
    digest_path = path.with_suffix('.sha256')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and digest_path.is_file():
        verify_archive_chunk(directory, receipt)
        try:
            verify_archive_chunk(path.parent, dict(file=path.name, sha256=digest_path.read_text().strip()))
            payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            if (payload['source'] == receipt['sha256'] and payload['contract'] == identity
                    and len(payload['tensors']['versions']) == receipt['ticks']):
                return payload['tensors'], True
        except (OSError, EOFError, ValueError, KeyError, RuntimeError, pickle.UnpicklingError):
            # Rebuild only the derived cache using the verified native source.
            pass
    with read_archive_rows(directory, receipt, scratch=scratch) as rows:
        tensors = pack_archive_rows(rows, schema)
    size = sum(value.nbytes for value in tensors['data'].values())
    try:
        require_archive_space(path.parent, size)
    except OSError:
        return tensors, False
    with tempfile.TemporaryDirectory(dir=path.parent) as pending:
        tmp = Path(pending) / path.name
        torch.save(dict(source=receipt['sha256'], contract=identity, tensors=tensors), tmp)
        with tmp.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        marker = Path(pending) / digest_path.name
        marker.write_text(digest + '\n')
        tmp.replace(path)
        marker.replace(digest_path)
    return tensors, False


def append_packed_archive(replay, payload):
    """Write a verified chunk in one ring operation, leaving native starts untouched."""
    if getattr(replay, 'recovery_failure_prefixes', False):
        raise ValueError('Recovery failure-prefix reconstruction requires full native rows')
    if payload['data'].keys() != replay.data.keys() or payload['approach'].keys() != replay.approach_metadata.keys():
        raise ValueError('Derived Replay tensor schema changed')
    ticks = len(payload['versions'])
    if not ticks or payload['excitation'].shape != (ticks, replay.lanes):
        raise ValueError('Derived Replay lane layout changed')
    first = max(0, ticks - replay.capacity)
    order = torch.arange(first, ticks)
    slots = (replay.write_pos + order) % replay.capacity
    data = {key: value[first:] for key, value in payload['data'].items()}
    for key, value in data.items():
        if value.shape != (len(slots), *replay.data[key].shape[1:]):
            raise ValueError(f'Derived Replay field shape changed for {key}')
        replay.data[key][slots] = value.to(dtype=replay.data[key].dtype)
    for key, value in payload['approach'].items():
        replay.approach_metadata[key][slots] = value[first:]
    replay.stage_ids[slots] = _stage_ids_from_tensors(data, replay.schema)
    replay.source_ids[slots] = torch.where(payload['excitation'][first:],
                                          SOURCE_CODES['action_identification'], SOURCE_CODES[replay.current_source])
    replay.policy_versions[slots] = payload['versions'][first:, None].expand(-1, replay.lanes)
    replay.streams[slots] = replay.current_stream
    replay.native_ticks[slots] = replay._native_tick_counter + order
    replay._native_tick_counter += ticks
    replay.write_pos = (replay.write_pos + ticks) % replay.capacity
    replay.count = min(replay.capacity, replay.count + ticks)
    replay._ordered_cache = None
