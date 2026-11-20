#!/usr/bin/env python3
"""Build the B1 2000-contact/1000-near-miss native substep pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quadsmash.rl.supermimic_contact_pilot_v1 import build_pilot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, type=Path,
                        help="Native substep .pt file or directory, repeatable")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--contacts", type=int, default=2000)
    parser.add_argument("--negatives", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1,
                        help="CPU trace readers. GPU simulation remains owned by its capture process")
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--force-threshold", type=float, default=.001)
    parser.add_argument("--near-miss-radius", type=int, default=4)
    args = parser.parse_args()
    if args.contacts < 0 or args.negatives < 0 or args.workers < 1:
        parser.error("contacts/negatives must be nonnegative and workers must be positive")
    manifest = build_pilot(args.input, args.output, contacts=args.contacts,
                           negatives=args.negatives, workers=args.workers,
                           seed=args.seed, force_threshold=args.force_threshold,
                           near_miss_radius=args.near_miss_radius)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
