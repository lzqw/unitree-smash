#!/usr/bin/env python3
"""Render a measured GPU-PhysX selected-case capture without Isaac RTX."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--training-index', type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f'Output already exists: {args.output}')
    required = [args.input / name for name in ('geometry.json', 'geometry.npz', 'link_poses.npy', 'trajectory.json', 'metadata.json')]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError('Offline capture is incomplete: ' + ', '.join(missing))
    metadata = json.loads((args.input / 'metadata.json').read_text())
    trace = json.loads((args.input / 'trajectory.json').read_text())
    events = metadata.get('event_records') or []
    event = events[-1] if events else {}
    best = {
        'contact_time_s': event.get('time_s'),
        'first_landing_time_s': event.get('landing_time_s'),
        'outgoing_velocity_mps': event.get('outgoing_velocity_mps'),
        'first_landing_position_m': event.get('landing_position_m'),
        'net_margin_m': event.get('net_margin_m'),
        'effective_face_contact': bool(event.get('effective_face_contact', False)),
        'complete_smash': bool(event.get('complete_smash', False)),
        'complete5': bool(event.get('complete5', False)),
        'robot_net_contact': bool(event.get('robot_net_contact', False)),
        'numerical_failures': int(event.get('numerical_failures', 0)),
        'recovery_success': bool(event.get('recovery_success', False)),
    }
    metrics = {
        'controller': 'supermimic_selected_case_offline_capture',
        'net_geometry_version': metadata.get('native_execution_contract', {}).get('net_geometry_version'),
        'episodes': 1,
        'complete': int(best['complete5'] or best['complete_smash']),
        'recovery': int(best['recovery_success']),
        'airborne_contact_count': int(any(row.get('contact', False) for row in trace)),
        'conditional_recovery_count': int(best['recovery_success']),
        'selected_video_lane': args.training_index,
        'formal_panel_recording': False,
        'presentation_title': 'SuperMimic selected case｜measured GPU PhysX replay',
        'presentation_subtitle': '真实GPU PhysX姿态与球轨迹｜CPU离线渲染，未重新模拟或插值',
        'presentation_population': '单案例开发视频',
        'presentation_conclusion': '视频用于动作诊断，不是新的正式评估',
        'best': best,
    }
    (args.input / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(ROOT / 'scripts' / 'render_g07_dynamic_video.py'),
        '--input', str(args.input), '--output', str(args.output), '--normal-only',
        '--fps', str(float(metadata.get('fps', 30))),
    ]
    return subprocess.run(cmd, cwd=str(ROOT), check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
