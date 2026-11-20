#!/usr/bin/env python3
"""Pack a completed native paired-probe collection for offline WM fitting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from quadsmash.rl.supermimic_action_identification_dataset_v1 import pack_collection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(pack_collection(args.collection, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
