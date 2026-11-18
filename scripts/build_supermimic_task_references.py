#!/usr/bin/env python3
"""Build terminal-verified v2 success bank on CPU; never start Isaac."""
import argparse,hashlib,json
from pathlib import Path
import torch
from quadsmash.rl.supermimic_full_body_geometry_v1 import FullBodyGeometryV1,MODEL
from quadsmash.rl.supermimic_task_space_reference_v1 import TaskSpaceReferenceBankV1
from quadsmash.rl.supermimic_data_split_v1 import explicit_lane_split

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packed-data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--policy-id',required=True);p.add_argument('--match-distance-threshold',type=float,required=True)
    p.add_argument('--episodes-summary',type=Path,required=True,help='Original independently scored terminal records, matching packed episode identities')
    p.add_argument('--split-manifest',type=Path,required=True,help='Whole-lane split; validation trajectories cannot enter Tracking')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    data=torch.load(a.packed_data,map_location='cpu',weights_only=True)
    _,heldout=explicit_lane_split(json.loads(a.split_manifest.read_text()),data['tensors']['episode'].shape[1])
    asset=json.loads(MODEL.read_text())['asset_sha256'];geometry=FullBodyGeometryV1(asset)
    with torch.no_grad():bank=TaskSpaceReferenceBankV1.from_packed(data['tensors'],geometry,
        source_sha256=hashlib.sha256(a.packed_data.read_bytes()).hexdigest(),policy_id=a.policy_id,
        match_distance_threshold=a.match_distance_threshold,terminal_records=json.loads(a.episodes_summary.read_text()),
        excluded_lanes=heldout.tolist())
    if not bank.entries:raise ValueError('No complete verified success reference; do not enable tracking')
    a.output.parent.mkdir(parents=True,exist_ok=True);torch.save(bank.payload(),a.output)
    a.output.with_suffix('.json').write_text(json.dumps(bank.manifest,indent=2)+'\n')
    print(json.dumps(bank.manifest))
if __name__=='__main__':main()
