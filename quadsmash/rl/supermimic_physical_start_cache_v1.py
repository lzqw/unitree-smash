"""Hash-verified current states; no archived future labels in imagination."""
import hashlib,json
from pathlib import Path
import torch
from .supermimic_physical_loop_replay_v1 import current_start
from .supermimic_physical_run_guard_v1 import atomic_torch_save

APPROACH_TICKS=(224,240,256,272)
# Recovery must cover the touchdown-to-recovery handoff.  Failures can occur
# shortly after 4s, so starting only at 432+ misses the first corrective
# window.  Keep one pre-4s touchdown transition plus early and late recovery.
RECOVERY_TICKS=(384,400,416,448,480,576,672)
# Overlap 64-step windows by 16 steps and cover the 800-step terminal.
CRITIC_TICKS=tuple(reversed(range(0,800,48)))


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
        for row in saved['starts'].values():
            if set(row)!={'before','executor_before','observation'}:
                raise ValueError('Future labels leaked into imagination start cache')
        return saved['starts']
    starts={};offset=0
    for entry in manifest['chunks']:
        source=archive/entry['file']
        if hashlib.sha256(source.read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError('Imagination-start source chunk hash mismatch')
        rows=torch.load(source,map_location='cpu',weights_only=True)['rows']
        for i,row in enumerate(rows):
            if offset+i in required:starts[offset+i]=current_start(row)
        offset+=len(rows)
        del rows
    if sorted(starts)!=required:raise ValueError('Incomplete full-episode current-state coverage')
    atomic_torch_save(dict(archive_signature=signature,starts=starts),path)
    return starts
