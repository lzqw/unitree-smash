#!/usr/bin/env python3
"""Compare commanded face normals on saved native TRAIN states, without dynamics."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quadsmash.control.full_racket_rally_controller_v4 import FullRacketRallyControllerV4


def angle(a, b):
    cosine = (a*b).sum(1) / (torch.linalg.vector_norm(a, dim=1)*torch.linalg.vector_norm(b, dim=1))
    return torch.rad2deg(torch.acos(cosine.clamp(-1, 1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['snapshot', 'baseline-checkpoint', 'counterfactual-checkpoint', 'joint-contract', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh output directory')
    torch.set_num_threads(2)
    baseline = torch.load(args.baseline_checkpoint, map_location='cpu', weights_only=True)
    alternative = torch.load(args.counterfactual_checkpoint, map_location='cpu', weights_only=True)
    assert torch.equal(baseline['parameters'], alternative['parameters'])
    assert baseline['seed'] == {k: v for k, v in alternative['seed'].items() if k != 'measured_face_feedback'}
    state_path = args.snapshot/'partial_search_state.pt'
    state = torch.load(state_path, map_location='cpu', weights_only=True)
    public_path = args.snapshot/'near_contact_all.partial.npz'
    control_path = args.snapshot/'control_window.partial.npz'
    plan_path = args.snapshot/'causal_plans.partial.json'
    contract = json.loads(args.joint_contract.read_text())
    names = [f'{leg}_{joint}_joint' for joint in ['hip', 'thigh', 'calf'] for leg in ['FL', 'FR', 'RL', 'RR']]
    names += [f'joint{i}' for i in range(1, 7)]
    defaults = contract['defaults_before_neutral_arm_override']
    defaults.update({f'joint{i+1}': q for i, q in enumerate(json.loads((ROOT/'configs/full_racket_reset_v4.json').read_text())['arm_joint_positions_rad'])})
    default = torch.tensor([defaults[n] for n in names])
    limits = torch.tensor([contract['limits'][n] for n in names])
    with np.load(public_path) as saved:
        count = saved['observations'].shape[1]
    cases = count // len(state['parameters'])
    controllers = [FullRacketRallyControllerV4(warm, count, 'cpu') for warm in [baseline, alternative]]
    for controller in controllers:
        controller.reset()
        controller.parameters.copy_(state['parameters'].repeat_interleave(cases, dim=0))
        controller.planned.fill_(True)
        for log in json.loads(plan_path.read_text()):
            for i, env in enumerate(log['env_ids']):
                event = log['calibrated_event_time_s'][i]
                if log['valid'][i] and np.isfinite(event) and 2.3 <= event <= 3.3:
                    controller.event_time[env] = event
                    controller.target_offset_world[env] = torch.tensor(np.array(log['position_m'][i])-controller.nominal_target)
    values, commands = [], []
    with np.load(public_path) as saved, np.load(control_path) as control, torch.inference_mode():
        for i, observed in enumerate(saved['observations']):
            obs = torch.tensor(observed)
            valid = saved['valid_first_episode'][i] & (observed[:, 773] < .5)
            assert np.allclose(saved['episode_clock_s'][i, valid], control['clock_s'][2*i, valid], atol=1e-5)
            actions = [c.action({'policy': obs}, default, limits) for c in controllers]
            assert all(torch.isfinite(a).all() and (a.abs() <= 1).all() for a in actions)
            joints = [limits[12:, 0]+.5*(a[:, 12:]+1)*(limits[12:, 1]-limits[12:, 0]) for a in actions]
            native = limits[12:, 0]+.5*(torch.tensor(control['actions'][2*i, :, 12:])+1)*(limits[12:, 1]-limits[12:, 0])
            teacher = controllers[0].teacher
            forward, up = obs[:, 763:766], -obs[:, 6:9]
            left = torch.linalg.cross(up, forward)
            normal = teacher.normal_world
            target = teacher.face_sign*(normal[0]*forward+normal[1]*left+normal[2]*up)
            before, after = [teacher.kinematics(q, teacher.contact_offset) for q in joints]
            actual = default[12:]+obs[:, 21:27]
            observed_normal = teacher.kinematics(actual, teacher.contact_offset)[1]
            values.append(torch.stack((angle(before[1], target), angle(after[1], target),
                angle(observed_normal, target), torch.linalg.vector_norm(after[0]-before[0], dim=1),
                (joints[0]-native).abs().amax(1), torch.tensor(valid)), dim=1).numpy())
            commands.append(torch.stack(actions).numpy())
    data = np.stack(values)
    valid = data[:, :, 5] > .5
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'counterfactual_native_state_commands.npz', diagnostics=data, actions=np.stack(commands))
    def stats(column):
        value = data[:, :, column][valid]
        return dict(minimum=float(value.min()), median=float(np.median(value)), maximum=float(value.max()))
    sources = [args.baseline_checkpoint, args.counterfactual_checkpoint, args.joint_contract,
        state_path, public_path, control_path, plan_path, Path(__file__),
        ROOT/'quadsmash/control/full_racket_teacher_v4.py', ROOT/'quadsmash/control/full_racket_face_feedback_v4.py']
    report = dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='Counterfactual full-teacher joint commands on original CEM13public native states and original causal forecasts. No changed-state dynamics, contact, recovery, or native avoidance outcome.',
        valid_unhit_samples=int(valid.sum()), observed_face_error_deg=stats(2),
        original_command_face_error_deg=stats(0), counterfactual_command_face_error_deg=stats(1),
        fraction_command_face_error_reduced=float(np.mean(data[:, :, 1][valid] < data[:, :, 0][valid]-.1)),
        maximum_command_point_displacement_m=stats(3), baseline_cpu_native_command_error_rad=stats(4),
        all_actions_finite_and_within_original_hardware_target_limits=True,
        new_dynamics_executed=False, formal_acceptance=False,
        sources={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (args.output/'REPORT.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
