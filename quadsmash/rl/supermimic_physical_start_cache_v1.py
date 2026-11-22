"""Hash-verified current states; no archived future labels in imagination."""
import hashlib,json
from pathlib import Path
import torch
from .supermimic_archive_storage_v1 import read_archive_rows
from .supermimic_physical_loop_replay_v1 import NATIVE_START_ID_FIELDS, current_start
from .supermimic_physical_run_guard_v1 import atomic_torch_save

APPROACH_TICKS=(224,240,256,272)
# Recovery must cover the touchdown-to-recovery handoff.  Failures can occur
# shortly after 4s, so starting only at 432+ misses the first corrective
# window.  Keep one pre-4s touchdown transition plus early and late recovery.
RECOVERY_TICKS=(384,400,416,448,480,576,672)
# Overlap 64-step windows by 16 steps and cover the 800-step terminal.
CRITIC_TICKS=tuple(reversed(range(0,800,48)))


def training_start_mask(row, excluded_lanes, *, recovery=False):
    observation=row['observation']
    if isinstance(observation,dict):
        observation=observation['policy']
    n=len(observation)
    mask=row.get('_actor_lane_mask',torch.ones(n,dtype=torch.bool)).to(device=observation.device,dtype=torch.bool).reshape(-1).clone()
    if len(mask)!=n:
        raise ValueError('Actor start lane mask has the wrong length')
    if n==1 and '_lane_id' in row:
        if int(row['_lane_id']) in excluded_lanes:
            mask.zero_()
    elif len(excluded_lanes):
        mask[list(excluded_lanes)]=False
    if recovery:
        from .supermimic_policy_execution_v1 import recovery_gate
        mask &= recovery_gate(row['before']['task'],observation)
    return mask


def load_start_cache(archive,manifest):
    archive=Path(archive)
    required=sorted(set(APPROACH_TICKS+RECOVERY_TICKS+CRITIC_TICKS))
    signature=hashlib.sha256(json.dumps(manifest['chunks'],sort_keys=True).encode()).hexdigest()
    # v4 adds the early touchdown-to-recovery handoff; do not reuse older caches.
    path=archive.parent/'physical_current_starts_v4.pt'
    if path.exists():
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if saved['archive_signature']!=signature or sorted(saved['starts'])!=required:
            raise ValueError('Current-start cache belongs to a different archive')
        required_fields = {'before', 'executor_before', 'observation'}
        allowed_fields = required_fields.union(NATIVE_START_ID_FIELDS)
        for row in saved['starts'].values():
            if not required_fields <= row.keys() <= allowed_fields:
                raise ValueError('Missing current state or unexpected fields in imagination start cache')
        return saved['starts']
    starts={};offset=0
    for entry in manifest['chunks']:
        with read_archive_rows(archive, entry, scratch=archive.parent) as rows:
            for i, row in enumerate(rows, start=offset):
                if i in required:
                    starts[i] = current_start(row)
            offset += len(rows)
    if sorted(starts)!=required:raise ValueError('Incomplete full-episode current-state coverage')
    atomic_torch_save(dict(archive_signature=signature,starts=starts),path)
    return starts
