#!/usr/bin/env python3
"""Compare disabled face-feedback mode to an exact original native teacher."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import types
import zipfile
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quadsmash.control.full_racket_teacher_v4 import FullRacketTeacherV4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'checkpoint', 'public', 'joint-contract', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh output directory')
    torch.set_num_threads(2)
    source_zip = args.run/'source_snapshot.zip'
    module_path = 'quadsmash/control/full_racket_teacher_v4.py'
    with zipfile.ZipFile(source_zip) as archive:
        source = archive.read(module_path)
    previous = types.ModuleType('quadsmash.control.previous_face_disabled_teacher')
    previous.__package__ = 'quadsmash.control'
    exec(compile(source, str(source_zip)+'/'+module_path, 'exec'), previous.__dict__)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    assert len(checkpoint['parameters']) == 32 and 'measured_face_feedback' not in checkpoint['seed']
    contract = json.loads(args.joint_contract.read_text())
    names = [f'{leg}_{joint}_joint' for joint in ['hip', 'thigh', 'calf'] for leg in ['FL', 'FR', 'RL', 'RR']]
    names += [f'joint{i}' for i in range(1, 7)]
    defaults = contract['defaults_before_neutral_arm_override']
    defaults.update({f'joint{i+1}': q for i, q in enumerate(json.loads((ROOT/'configs/full_racket_reset_v4.json').read_text())['arm_joint_positions_rad'])})
    default = torch.tensor([defaults[n] for n in names])
    limits = torch.tensor([contract['limits'][n] for n in names])
    values = []
    for width in [26, 32]:
        teachers = [cls(checkpoint['backbone'], checkpoint['seed'], ROOT/'configs/g07_go2_walk_transfer.json', 2, 'cpu')
                    for cls in [previous.FullRacketTeacherV4, FullRacketTeacherV4]]
        for teacher in teachers:
            teacher.reset()
        parameters = checkpoint['parameters'][:width][None].repeat(2, 1)
        commands = []
        with np.load(args.public) as recorded, torch.inference_mode():
            for i, obs in enumerate(recorded['observations']):
                if not recorded['valid_first_episode'][i].all():
                    break
                before, after = [t.action({'policy': torch.tensor(obs)}, parameters, default, limits) for t in teachers]
                assert torch.equal(before, after), (width, i)
                commands.append(after.numpy())
        values.append(np.stack(commands))
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'legacy_commands.npz', actions=np.stack(values))
    sources = [source_zip, args.checkpoint, args.public, args.joint_contract, Path(__file__), ROOT/module_path]
    report = dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='CPU comparison against exact original native teacher on recorded public inputs, with measured-face mode disabled.',
        parameter_widths=[26, 32], lanes=2, ticks_per_mode=len(values[0]),
        all_teacher_commands_bitwise_equal=True, legacy_teacher_sha256=hashlib.sha256(source).hexdigest(),
        new_dynamics_executed=False, formal_acceptance=False,
        sources={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (args.output/'REPORT.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
