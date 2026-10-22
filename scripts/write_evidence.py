#!/usr/bin/env python3
"""Create observed evidence records from actual run artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--artifacts", nargs="+", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--gpu-hours", type=float, default=0.0)
    parser.add_argument("--env-steps", type=int, default=0)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    contract = root / "configs/acceptance.json"
    contract_sha = digest(contract)
    definitions = json.loads(contract.read_text(encoding="utf-8"))["goals"]
    definition = definitions[args.goal]
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    if "checks" in metrics:
        metrics = metrics["checks"]
    artifacts = []
    for value in args.artifacts:
        path = Path(value)
        if path.is_absolute():
            try:
                relative = path.resolve().relative_to(root.resolve())
            except ValueError as exc:
                raise ValueError(f"artifact is outside project: {path}") from exc
        else:
            relative = path
            path = root / path
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        artifacts.append({"path": str(relative), "sha256": digest(path), "bytes": path.stat().st_size})
    started = datetime.now(timezone.utc)
    evidence = {
        "schema_version": 1,
        "goal_id": args.goal,
        "task_id": definition["task_id"],
        "contract_sha256": contract_sha,
        "evidence_kind": "observed",
        "run": {"command": args.command, "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True).stdout.strip() or "no-commit", "started_at": started.isoformat(), "finished_at": datetime.now(timezone.utc).isoformat(), "gpu_hours": args.gpu_hours, "env_steps": args.env_steps},
        "metrics": metrics,
        "artifacts": artifacts,
    }
    output = root / f"reports/{args.goal}/evidence.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
