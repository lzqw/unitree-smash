"""Transfer a frozen Recovery head and its release-time arm reference to an Approach checkpoint."""
import argparse
import hashlib
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

from quadsmash.rl.supermimic_physical_loop_loading_v1 import RECOVERY_NAMES
from scripts.run_supermimic_recovery_baseline import verify


def merged_state(approach, recovery):
    state = dict(approach)
    if sum(recovery[key].numel() for key in RECOVERY_NAMES) != 111186:
        raise ValueError('Recovery parameter count differs')
    for key in RECOVERY_NAMES:
        ref = key.replace('actor.recovery.', 'release_arm_reference.')
        if ref not in recovery:
            raise ValueError('Recovery checkpoint lacks its frozen 40 ms reference')
        for target_key, like in ((key, approach[key]), (ref, approach.get(ref, approach[key]))):
            value = recovery[target_key]
            if value.shape != like.shape or value.dtype != like.dtype or not torch.isfinite(value).all():
                raise ValueError('Incompatible Recovery weight: ' + target_key)
            if target_key == ref and not torch.equal(value, like):
                raise ValueError('Early release actions differ from target Approach baseline')
            state[target_key] = value
    return state


def merge(approach_path, recovery_path, output):
    verify([approach_path, recovery_path], synchronize_recovery=True)
    approach, recovery = [torch.load(p, mmap=True, map_location='cpu', weights_only=False)
                          for p in (approach_path, recovery_path)]
    inputs = {k: hashlib.sha256(p.read_bytes()).hexdigest()
              for k, p in [('approach', approach_path), ('recovery', recovery_path)]}
    result = dict(approach, actor=merged_state(approach['actor'], recovery['actor']),
                  metadata=dict(approach['metadata'], recovery_merge_sha256=inputs))
    with NamedTemporaryFile(dir=output.parent, suffix='.pt') as tmp:
        torch.save(result, tmp.name)
        # Publish once, including when concurrent evaluations request the same path.
        os.link(tmp.name, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('approach', 'recovery', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    merge(args.approach, args.recovery, args.output)
