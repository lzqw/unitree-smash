#!/usr/bin/env python3
"""Compare optional standing recovery coordinates with the exact native teacher."""
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
from quadsmash.control.full_racket_recovery_search_v4 import extend_recovery_parameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'checkpoint', 'public', 'joint-contract', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh output directory')
    torch.set_num_threads(2)
    module_path = 'quadsmash/control/full_racket_teacher_v4.py'
    with zipfile.ZipFile(args.run/'source_snapshot.zip') as z:
        source = z.read(module_path)
    old = types.ModuleType('quadsmash.control.previous_standing_recovery_teacher')
    old.__package__ = 'quadsmash.control'
    exec(compile(source, str(args.run/'source_snapshot.zip')+'/'+module_path, 'exec'), old.__dict__)
    warm = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    assert len(warm['parameters']) == 32
    contract = json.loads(args.joint_contract.read_text())
    names = [f'{leg}_{joint}_joint' for joint in ['hip','thigh','calf'] for leg in ['FL','FR','RL','RR']]
    names += [f'joint{i}' for i in range(1,7)]
    defaults = contract['defaults_before_neutral_arm_override']
    defaults.update({f'joint{i+1}':q for i,q in enumerate(json.loads((ROOT/'configs/full_racket_reset_v4.json').read_text())['arm_joint_positions_rad'])})
    default = torch.tensor([defaults[n] for n in names])
    limits = torch.tensor([contract['limits'][n] for n in names])
    all_values, variants = [], []
    for handover in [float(warm['parameters'][3]), 1.30]:
        teachers = [cls(warm['backbone'], warm['seed'], ROOT/'configs/g07_go2_walk_transfer.json', 2, 'cpu')
                    for cls in [old.FullRacketTeacherV4, FullRacketTeacherV4, FullRacketTeacherV4, FullRacketTeacherV4]]
        for teacher in teachers:
            teacher.reset()
        before = warm['parameters'].clone()
        before[3] = handover
        extended = extend_recovery_parameters(before, warm['backbone']['recovery_parameters'])
        changed = extended.clone()
        changed[36] = .8
        parameters = [p[None].repeat(2, 1) for p in [before, before, extended, changed]]
        values, recovered_ticks, changed_ticks = [], 0, 0
        with np.load(args.public) as saved, torch.inference_mode():
            for i, observed in enumerate(saved['observations']):
                if not saved['valid_first_episode'][i].all():
                    break
                obs = torch.tensor(observed)
                commands = [teacher.action({'policy':obs}, p, default, limits) for teacher,p in zip(teachers, parameters)]
                assert torch.equal(commands[0], commands[1]) and torch.equal(commands[0], commands[2]), (handover, i)
                assert all(torch.isfinite(a).all() and (a.abs()<=1).all() for a in commands)
                local = (obs[:,775]-before[2])*before[4]
                inactive = local < handover
                assert torch.equal(commands[2][inactive], commands[3][inactive]), (handover,i)
                assert torch.equal(commands[2][:,12:], commands[3][:,12:]), (handover,i)
                recovered_ticks += int((~inactive).any())
                changed_ticks += int(not torch.equal(commands[2], commands[3]))
                values.append(torch.stack(commands).numpy())
        if handover == 1.30:
            assert recovered_ticks > 10 and changed_ticks > 0
        all_values.append(np.stack(values))
        variants.append(dict(local_handover_time_s=handover, ticks=len(values),
            original32_new32_and_default37_commands_bitwise_equal=True,
            altered_recovery_gain_only_changes_legs_after_handover=True,
            ticks_with_recovery_active=recovered_ticks, ticks_with_changed_leg_commands=changed_ticks))
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'counterfactual_recovery_commands.npz', actions=np.stack(all_values))
    paths = [args.run/'source_snapshot.zip',args.checkpoint,args.public,args.joint_contract,Path(__file__),
        ROOT/module_path,ROOT/'quadsmash/control/full_racket_recovery_search_v4.py',ROOT/'quadsmash/control/full_racket_arm_search_v4.py']
    report = dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='CPU replay on identical recorded public observations. Includes deliberately earlier handover to exercise standing-controller compatibility. No changed-state dynamics or recovery outcome.',
        lanes=2, variants=variants, legacy_teacher_sha256=hashlib.sha256(source).hexdigest(),
        native_execution=False, formal_acceptance=False,
        source_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (args.output/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'}),flush=True)


if __name__ == '__main__':
    main()
