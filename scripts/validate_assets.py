#!/usr/bin/env python3
"""Run native G01 revalidation against the standard asset, charging G04 repair budget."""
import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("reports/G01"))
    args = parser.parse_args()
    if args.num_envs < 2 or args.num_envs % 2 or args.episodes < 100:
        parser.error("paired native audit requires an even num-envs >= 2 and episodes >= 100")
    root = Path(__file__).resolve().parents[1]
    try:
        output = args.output.resolve().relative_to(root)
    except ValueError:
        parser.error("output must be inside the project for evidence provenance")
    python = "/home/autodrive/Isaac/env_isaaclab/bin/python"
    version = datetime.now(timezone.utc).strftime("native_%Y%m%dT%H%M%S_%f")
    base = output / version
    stand, calibration = base / "stand", base / "calibration"
    asset = root / "assets/quadsmash_standard_v2.usd"
    if not asset.is_file():
        parser.error("the source-verified standard asset has not been built")
    jobs = [
        (stand, 240, (math.ceil(args.episodes / args.num_envs) * 1000 + 51) * args.num_envs, "metrics.json",
         [python, "-u", "scripts/isaac_asset_reaudit.py", "--asset", str(asset), "--output", str(stand),
          "--num-envs", str(args.num_envs), "--episodes", str(args.episodes), "--self-collisions", "--headless"]),
        (calibration, 120, 25200, "calibration.json",
         [python, "-u", "scripts/isaac_asset_calibration.py", "--asset", str(asset), "--output", str(calibration), "--self-collisions", "--headless"]),
    ]
    for destination, seconds, steps, receipt, command in jobs:
        guarded = [sys.executable, "scripts/run_gpu_job.py", "--goal", "G04", "--max-seconds", str(seconds),
                   "--env-steps-cap", str(steps), "--output", str(destination), "--require-artifact", receipt, "--", *command]
        subprocess.run(guarded, cwd=root, check=True)
    subprocess.run([python, "scripts/finalize_g01_reaudit.py", "--stand-dir", str(stand),
                    "--calibration-dir", str(calibration), "--flight-dir", str(stand), "--output", str(output)], cwd=root, check=True)


if __name__ == "__main__":
    main()
