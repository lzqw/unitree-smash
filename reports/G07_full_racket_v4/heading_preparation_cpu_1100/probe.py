"""Replay an isolated heading-feedback draft without changing live job sources."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import quadsmash.control.full_racket_rally_controller_v4 as rally
from quadsmash.control.initial_intercept import observed_world_heading


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    output = Path(__file__).resolve().parent
    assert not (output / 'REPORT.json').exists()
    source_path = ROOT / 'quadsmash/control/full_racket_teacher_v4.py'
    original = source_path.read_text()
    draft = original.replace('import torch\n', 'import torch\nfrom math import isfinite\nfrom .initial_intercept import observed_world_heading\n', 1)
    marker = '        self.smooth_recovery_arm=seed.get(\'smooth_recovery_arm\',False)\n'
    assert draft.count(marker) == 1
    draft = draft.replace(marker, marker + '''        raw_heading_gain=seed.get('walk_heading_gain',0.)
        if isinstance(raw_heading_gain,bool) or not isinstance(raw_heading_gain,(int,float)):
            raise ValueError('Walk heading gain must be a finite number in[0,4]')
        self.walk_heading_gain=float(raw_heading_gain)
        if not isfinite(self.walk_heading_gain) or not 0.<=self.walk_heading_gain<=4.:
            raise ValueError('Walk heading gain must be a finite number in[0,4]')
''', 1)
    marker = "        walk=self.walk.action({'policy':walk_obs},walk_parameters,default,limits,maximum_time=3.)"
    assert draft.count(marker) == 1
    draft = draft.replace(marker, '''        heading_arguments={}
        if self.walk_heading_gain:
            # The known task direction is world +X, expressed in the public
            # body frame. Gravity supplies the second orientation reference.
            heading=observed_world_heading(obs,[1.,0.,0.])
            heading_arguments=dict(heading_error=-heading,
                heading_kp=torch.full_like(time,self.walk_heading_gain))
        walk=self.walk.action({'policy':walk_obs},walk_parameters,default,limits,
            maximum_time=3.,**heading_arguments)''', 1)
    proposed = output / 'proposed_teacher.py'
    proposed.write_text(draft)
    module = types.ModuleType('quadsmash.control._heading_draft_teacher')
    module.__file__ = str(proposed)
    exec(compile(draft, str(proposed), 'exec'), module.__dict__)
    checkpoint_path = ROOT / 'reports/G07_full_racket_v4/cem32_lateral_feedback_1015/best_candidate.pt'
    public_path = ROOT / 'reports/G07_full_racket_v4/cem31_candidate5_diag_0900/run/public_and_actions_first2.npz'
    pose_path = public_path.with_name('all_first_episodes.npz')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    assert checkpoint['native_execution'] and 'walk_heading_gain' not in checkpoint['seed']
    original_class = rally.FullRacketTeacherV4
    controllers = []
    for teacher, gain in [(original_class, None), (module.FullRacketTeacherV4, None),
                          (module.FullRacketTeacherV4, 0.), (module.FullRacketTeacherV4, 1.5)]:
        variant = copy.deepcopy(checkpoint)
        if gain is not None:
            variant['seed']['walk_heading_gain'] = gain
        rally.FullRacketTeacherV4 = teacher
        try:
            controllers.append(rally.FullRacketRallyControllerV4(variant, 2, 'cpu'))
        finally:
            rally.FullRacketTeacherV4 = original_class
    contract_path = ROOT / 'reports/G07_full_racket_v4/CPU_REPLAY_AUTHORED_JOINTS_0724.json'
    contract = json.loads(contract_path.read_text())
    names = [f'{leg}_{joint}_joint' for joint in ['hip','thigh','calf'] for leg in ['FL','FR','RL','RR']]+[f'joint{i}' for i in range(1,7)]
    defaults = dict(contract['defaults_before_neutral_arm_override'])
    arm = json.loads((ROOT/'configs/full_racket_reset_v4.json').read_text())['arm_joint_positions_rad']
    defaults.update({f'joint{i+1}':v for i,v in enumerate(arm)})
    default = torch.tensor([defaults[n] for n in names], dtype=torch.float32)
    limits = torch.tensor([contract['limits'][n] for n in names], dtype=torch.float32)
    streams = [[] for _ in controllers]
    with np.load(public_path) as data:
        observations = data['observations'].copy()
        validity = data['valid_first_episode'].copy()
    with torch.inference_mode():
        for obs, valid in zip(observations, validity):
            if not valid.all():
                break
            public = torch.tensor(obs)
            before = public.clone()
            for controller, stream in zip(controllers, streams):
                action = controller.action({'policy':public}, default, limits)
                assert action.shape == (2,18) and torch.isfinite(action).all() and (abs(action)<=1).all()
                assert torch.equal(public, before)
                stream.append(action.numpy().copy())
    arrays = [np.stack(s) for s in streams]
    assert len(arrays[0]) == 800
    assert np.array_equal(arrays[0],arrays[1]) and np.array_equal(arrays[0],arrays[2])
    assert np.array_equal(arrays[0][:,:,12:],arrays[3][:,:,12:])
    leg_changes = np.any(arrays[0][:,:,:12] != arrays[3][:,:,:12], axis=(1,2))
    assert leg_changes.any() and not leg_changes[230:].any()
    with np.load(pose_path) as native:
        q = native['link_poses'][:,:2,0,3:]
        w,x,y,z = np.moveaxis(q,-1,0)
        measured_yaw = np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
        recovered = observed_world_heading(torch.tensor(observations[::2].reshape(-1,794)), [1.,0.,0.]).numpy().reshape(-1,2)
        error = np.arctan2(np.sin(recovered-measured_yaw),np.cos(recovered-measured_yaw))
        assert np.max(abs(error)) < 2e-5
    report = dict(scope='Isolated CPU draft replay only; not installed in live training sources and not physically executed.',
        native_execution=False, formal_acceptance=False, performance_inherited=False,
        gain=1.5, replay_ticks=len(arrays[0]), default_and_zero_bitwise_legacy=True,
        changed_leg_ticks=int(leg_changes.sum()), changed_arm_ticks=0,
        changed_commands_only_before_control_tick=230,
        heading_public_vs_native_max_error_rad=float(abs(error).max()),
        observation_immutable=True, new_dynamics_or_asset_changes=False,
        caveat='Identical arm commands apply only to frozen-input replay; changing walking can change arm/body motion in closed-loop dynamics.',
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in [source_path, proposed, checkpoint_path, public_path, pose_path, contract_path, Path(__file__)]})
    (output/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
