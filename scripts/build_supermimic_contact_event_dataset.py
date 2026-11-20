#!/usr/bin/env python3
"""Build event-aligned contact windows from a packed native replay."""

import argparse
import json
from pathlib import Path

from quadsmash.rl.supermimic_contact_event_dataset_v1 import build_from_packed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packed-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--negative-ratio", type=int, default=2)
    parser.add_argument("--max-negative", type=int, default=None)
    parser.add_argument("--lane-split-manifest", type=Path, default=None,
                        help="Explicit disjoint train/validation/holdout whole-lane split")
    parser.add_argument("--pre-control-ticks", type=int, default=4,
                        help="Number of control ticks before the event center")
    parser.add_argument("--post-control-ticks", type=int, default=64,
                        help="Number of control ticks after the event center")
    args = parser.parse_args()
    manifest = build_from_packed(args.packed_data, args.output,
                                  negative_ratio=args.negative_ratio,
                                  max_negative=args.max_negative,
                                  lane_split_manifest=args.lane_split_manifest,
                                  pre_control_ticks=args.pre_control_ticks,
                                  post_control_ticks=args.post_control_ticks)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
