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
    args = parser.parse_args()
    manifest = build_from_packed(args.packed_data, args.output,
                                  negative_ratio=args.negative_ratio,
                                  max_negative=args.max_negative)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
