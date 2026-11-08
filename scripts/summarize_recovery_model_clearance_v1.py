"""Describe sampled head clearance and touchdown state of model candidates.

Surface geometry is the authored collision geometry. These 10 ms samples do
not replace the substep path check or native paired ground-contact evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.pure_control_long_trajectory_recovery_v11 import PureControlLongTrajectoryRecoveryV11
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.event_recovery_manager_v1 import authored_clearance_state
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['configuration', 'fixture-run', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--model-run', type=Path, action='append', required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Fresh evidence required')
    read = lambda p: json.loads(Path(p).read_text())
    cfg = read(args.configuration)
    nominal = read(args.fixture_run / 'native_command_inputs.json')['default_joint_position_rad']
    c = PureControlLongTrajectoryRecoveryV11(PositionPDInterface(**cfg['interface']), nominal, cfg['model_reference_library'])
    paths = [p for directory in args.model_run for p in sorted(directory.glob('case_*.json'))]
    records = []
    for path in paths:
        result = read(path)
        points = []
        for row in result['trajectory']:
            state = c.library.state(dict(query=row), -1)
            surfaces = authored_clearance_state(c.geometry, state)
            head = min((dict(body=name, **surfaces[name]) for name in ['Head_lower', 'Head_upper']), key=lambda x: x['gap_m'])
            nonfoot = min((dict(body=name, **value) for name, value in surfaces.items()), key=lambda x: x['gap_m'])
            rotation = state.root_rotation_world
            points.append(dict(step=row['step'], after_release_s=(row['step'] + 1) * .01,
                head=head, lowest_nonfoot=nonfoot, contacts=state.contacts,
                root_velocity_world=rotation @ state.v[:3],
                root_angular_speed_rad_s=float(np.linalg.norm(state.v[3:6])),
                roll_pitch_rad=[np.arctan2(rotation[2, 1], rotation[2, 2]), np.arcsin(np.clip(-rotation[2, 0], -1, 1))],
                calf_joint_rad=state.joint_position[8:12], arm_joint_rad=state.joint_position[12:]))
        support = next((r for r in points if np.any(r['contacts'])), None)
        four_feet = next((r for r in points if np.all(r['contacts'])), None)
        records.append(dict(source_case=result['source_case'],
            sampled_minimum_head=min(points, key=lambda r: r['head']['gap_m']),
            sampled_minimum_nonfoot=min(points, key=lambda r: r['lowest_nonfoot']['gap_m']),
            first_sampled_support=support, first_sampled_four_feet=four_feet, final=points[-1],
            source_full_episode_model_recovery=result['model_episode_recovery']))
    pins = [Path(__file__), args.configuration, Path(cfg['model_reference_library']), *paths]
    output = dict(records=records, source_sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in pins},
        sample_period_s=.01, native_success=False, physical_collision_authority='native paired contact',
        limitation='Reports control-boundary samples, not the minimum over every substep, and not native clearance.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plain(output), indent=2, allow_nan=False) + '\n')
    print(json.dumps(plain(dict(cases=len(records), sampled_minimum_head_m={r['source_case']:r['sampled_minimum_head']['head']['gap_m'] for r in records}))))


if __name__ == '__main__':
    main()
