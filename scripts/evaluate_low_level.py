#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quadsmash.control import StandSkill, ArmSwingSkill, JumpRecoverySkill

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", default="reports/low_level.json"); args = parser.parse_args()
    stand = StandSkill(12).step(np.zeros(12), np.zeros(12))
    swing = ArmSwingSkill(6).step(1.0, np.zeros(6))
    recovery = JumpRecoverySkill().step(0.25, -0.4)
    result = {"stand": {"success": stand.success, **stand.metrics}, "arm_swing": {"success": swing.success, **swing.metrics}, "jump_recovery": {"success": recovery.success, **recovery.metrics}, "events": [stand.event, swing.event, recovery.event]}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps(result, indent=2))

if __name__ == "__main__": main()

