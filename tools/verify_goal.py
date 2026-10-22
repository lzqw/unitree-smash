#!/usr/bin/env python3
"""Validate goal evidence; never launch training or change goal status.

This is a structural/threshold audit, not proof that a simulation really ran.
Requires only the Python standard library. Exit 0: pass, 2: failed evidence,
3: invalid invocation/contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inside(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError('Artifact path must be a nonempty relative path.')
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError(f'Absolute path is not allowed: {relative}')
    path = (root / candidate).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f'Path escapes workspace: {relative}') from exc
    return path


def finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def check_metric(value: Any, spec: dict[str, Any]) -> bool:
    expected = spec.get('value')
    op = spec.get('op')
    if isinstance(expected, bool):
        return op == 'eq' and type(value) is bool and value is expected
    if not finite_number(value) or not finite_number(expected):
        return False
    if op == 'eq':
        return value == expected
    if op == 'ge':
        return value >= expected
    if op == 'le':
        return value <= expected
    raise ValueError(f'Unsupported metric operator: {op}')


def read_object(path: Path) -> dict[str, Any]:
    with path.open(encoding='utf-8') as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f'Expected JSON object: {path}')
    return value


def verify(root: Path, goal_id: str) -> dict[str, Any]:
    root = root.resolve()
    contract_path = inside(root, 'configs/acceptance.json')
    contract = read_object(contract_path)
    goals = contract.get('goals')
    if not isinstance(goals, dict) or goal_id not in goals:
        raise ValueError(f'Unknown goal {goal_id!r}.')
    contract_hash = digest(contract_path)
    results: dict[str, list[str]] = {}
    active: set[str] = set()

    def visit(gid: str) -> list[str]:
        if gid in results:
            return results[gid]
        if gid in active:
            raise ValueError(f'Cyclic goal dependency: {gid}')
        if gid not in goals:
            raise ValueError(f'Missing dependency in contract: {gid}')
        active.add(gid)
        definition = goals[gid]
        errors: list[str] = []
        for parent in definition.get('depends_on', []):
            if visit(parent):
                errors.append(f'Dependency did not pass: {parent}')
        report_path = inside(root, f'reports/{gid}/evidence.json')
        try:
            evidence = read_object(report_path)
        except (OSError, ValueError) as exc:
            errors.append(f'Missing or invalid evidence: {exc}')
            active.remove(gid)
            results[gid] = errors
            return errors

        for field, expected in (
            ('schema_version', 1), ('goal_id', gid),
            ('task_id', definition['task_id']),
            ('contract_sha256', contract_hash), ('evidence_kind', 'observed'),
        ):
            if evidence.get(field) != expected:
                errors.append(f'{field} must equal {expected!r}')
        run = evidence.get('run')
        if not isinstance(run, dict):
            errors.append('Missing run metadata object.')
        else:
            for field in ('command', 'git_commit', 'started_at', 'finished_at'):
                if not isinstance(run.get(field), str) or not run[field].strip():
                    errors.append(f'Missing nonempty run.{field}')
            for field in ('gpu_hours', 'env_steps'):
                value = run.get(field)
                if not finite_number(value) or value < 0:
                    errors.append(f'run.{field} must be a nonnegative finite number.')
            if not isinstance(run.get('env_steps'), int) or isinstance(run.get('env_steps'), bool):
                errors.append('run.env_steps must be an integer.')
            try:
                start = datetime.fromisoformat(run.get('started_at', '').replace('Z', '+00:00'))
                end = datetime.fromisoformat(run.get('finished_at', '').replace('Z', '+00:00'))
                if start.tzinfo is None or end.tzinfo is None or end < start:
                    errors.append('Timestamps must include timezone and be ordered.')
            except (ValueError, TypeError, AttributeError):
                errors.append('Run timestamps must be valid ISO-8601 strings.')

        metrics = evidence.get('metrics')
        if not isinstance(metrics, dict):
            errors.append('Missing metrics object.')
            metrics = {}
        for name, spec in definition.get('checks', {}).items():
            if name not in metrics:
                errors.append(f'Missing metric: {name}')
            elif not check_metric(metrics[name], spec):
                errors.append(f'Metric {name}={metrics[name]!r} does not meet {spec!r}')

        entries = evidence.get('artifacts')
        artifacts: dict[str, Any] = {}
        if not isinstance(entries, list):
            errors.append('artifacts must be a list.')
            entries = []
        for item in entries:
            if not isinstance(item, dict) or not isinstance(item.get('path'), str):
                errors.append('Invalid artifact entry.')
                continue
            name = item['path']
            if name in artifacts:
                errors.append(f'Duplicate artifact: {name}')
            artifacts[name] = item
            try:
                path = inside(root, name)
                if not path.is_file() or path.stat().st_size == 0:
                    errors.append(f'Missing or empty artifact: {name}')
                elif item.get('sha256') != digest(path):
                    errors.append(f'Artifact hash mismatch: {name}')
            except (OSError, ValueError) as exc:
                errors.append(str(exc))
        for required in definition.get('required_artifacts', []):
            if required not in artifacts:
                errors.append(f'Required artifact not listed: {required}')
        active.remove(gid)
        results[gid] = errors
        return errors

    visit(goal_id)
    return {
        'goal_id': goal_id,
        'status': 'PASSED' if not results[goal_id] else 'FAILED',
        'contract_sha256': contract_hash,
        'checks': {gid: {'passed': not errors, 'errors': errors} for gid, errors in results.items()},
        'scope': 'Structure, hashes, dependencies and thresholds only; not simulation authenticity or GPU budget enforcement.',
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--goal', required=True)
    args = parser.parse_args()
    try:
        result = verify(args.root, args.goal)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({'status': 'ERROR', 'error': str(exc)}, ensure_ascii=False, indent=2))
        return 3
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['status'] == 'PASSED' else 2


if __name__ == '__main__':
    sys.exit(main())
