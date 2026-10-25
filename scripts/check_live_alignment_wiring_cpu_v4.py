"""Execute actual reward methods on a CPU numerical wiring fixture.

Body poses and velocities come from completed native recordings. Rally flags,
torques and resets are controlled numerical fixtures: no Isaac constructor or
physics step is called, no historical rewards are reconstructed, and this is
not native execution or policy-performance acceptance.
"""
import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
from scipy.spatial.transform import Rotation
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quadsmash.physics.native import quat_rotate
from quadsmash.control.full_racket_residual_reward_v4 import ResidualEventRewardV4
from quadsmash.control.measured_joint_placement_reward_v4 import MeasuredJointPlacementRewardV4
from quadsmash.control.measured_string_alignment_reward_v4 import (
    MeasuredStringAlignmentRewardV4, attack_aligned_target)

MODULE = 'quadsmash/tasks/stand/full_racket_residual_env_v4.py'
TOLERANCES = dict(local_position_m=5e-5, plane_rate_mps=2e-3,
    independent_potential=2e-3, translated_reward=.01, payment_difference=3e-5)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CPUBaseFixture:
    def _reset_idx(self, ids):
        self.fixture_parent_reset_ids = ids.clone()


class InputRecorder(MeasuredStringAlignmentRewardV4):
    def step(self, **kwargs):
        self.fixture_inputs = {name:value.clone() for name,value in kwargs.items()}
        return super().step(**kwargs)


class ControlledRallyFlags:
    def __init__(self, count):
        for name in ['legal_initial_contact', 'string_hit', 'airborne_string_hit', 'released',
                'crossed_net', 'landed_target', 'ground_seen', 'had_contact', 'frame_hit']:
            setattr(self, name, torch.zeros(count, dtype=torch.bool))
        self.in_play = torch.ones(count, dtype=torch.bool)
        self.fixture_fault = torch.zeros(count, dtype=torch.bool)
        self.outgoing_velocity = torch.full((count, 3), torch.nan)
        self.landing_point = torch.full((count, 3), torch.nan)

    def faulted(self):
        return self.fixture_fault

    def robot_rule_faulted(self):
        return self.fixture_fault

    def complete(self, jump):
        return torch.zeros_like(jump)


def methods(source, label):
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
        and node.name == 'FullRacketResidualTrainingEnvV4')
    functions = [copy.deepcopy(node) for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in ['_get_rewards', '_reset_idx']]
    assert len(functions) == 2
    fixture = ast.ClassDef(name=label, bases=[ast.Name(id='CPUBaseFixture', ctx=ast.Load())],
        keywords=[], body=functions, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[fixture], type_ignores=[]))
    namespace = dict(torch=torch, quat_rotate=quat_rotate, attack_aligned_target=attack_aligned_target,
        CPUBaseFixture=CPUBaseFixture)
    exec(compile(module, '<actual reward methods in explicit CPU fixture>', 'exec'), namespace)
    return namespace[label]


def make_fixture(cls, poses, balls, root_rotation, public, settings, court, racket_id, *, origin=None, joint=True):
    count = len(public); obj = object.__new__(cls)
    origin = np.zeros((count, 3)) if origin is None else np.asarray(origin)
    p, b = np.asarray(poses), np.asarray(balls)
    root_w = np.einsum('nij,nj->ni', root_rotation, public[:, 3:6])
    root_v = np.einsum('nij,nj->ni', root_rotation, public[:, :3])
    tool_v = root_v+np.cross(root_w, p[:, racket_id, :3]-p[:, 0, :3])+np.einsum('nij,nj->ni', root_rotation, public[:, 51:54])
    tool_w = root_w+np.einsum('nij,nj->ni', root_rotation, public[:, 54:57])
    ball_v = root_v+np.cross(root_w, b-p[:, 0, :3])+np.einsum('nij,nj->ni', root_rotation, public[:, 64:67])
    velocities = np.zeros((count, p.shape[1], 3)); angular = np.zeros_like(velocities)
    velocities[:, racket_id] = tool_v; angular[:, racket_id] = tool_w
    obj.device = 'cpu'; obj.num_envs = count; obj.step_dt = .01; obj.racket_body_id = racket_id
    obj.scene = SimpleNamespace(env_origins=torch.tensor(origin, dtype=torch.float32))
    obj.robot = SimpleNamespace(data=SimpleNamespace(
        body_link_pos_w=torch.tensor(p[:, :, :3]+origin[:, None], dtype=torch.float32),
        body_link_quat_w=torch.tensor(p[:, :, 3:], dtype=torch.float32),
        body_link_lin_vel_w=torch.tensor(velocities, dtype=torch.float32),
        body_link_ang_vel_w=torch.tensor(angular, dtype=torch.float32),
        applied_torque=(torch.arange(count*18).reshape(count,18)%21-10).float()))
    obj.shuttle = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.tensor(b+origin,dtype=torch.float32),
        root_lin_vel_w=torch.tensor(ball_v,dtype=torch.float32)))
    obj.rally = ControlledRallyFlags(count)
    obj.residual_rewards = ResidualEventRewardV4(count, 'cpu')
    obj.joint_placement_reward = MeasuredJointPlacementRewardV4(count, 'cpu', court,
        dict(weight=20.,distance_scale_m=.25)) if joint else None
    obj.live_alignment_reward = (InputRecorder(count, 'cpu', settings,
        semiaxes_yz_m=[.0915,.1165], contact_plane_offset_m=.0256) if settings is not None else None)
    for name in ['_episode_has_stepped', 'rewarded_jump', 'rewarded_success', 'rewarded_body_contact',
            'flight_qualified', 'jump_success', 'body_ground_seen', 'reset_time_outs', 'reset_terminated']:
        setattr(obj, name, torch.zeros(count,dtype=torch.bool))
    obj.episode_length_buf = torch.tensor(np.rint(public[:,775]/.01),dtype=torch.long)
    obj.reward_event_time = torch.full((count,),2.75)
    obj.reward_plan_valid = torch.ones(count,dtype=torch.bool)
    obj.reward_target = torch.tensor([1.1,0.,1.8]).expand(count,-1).clone()
    obj.reward_velocity = torch.tensor([1.5,0.,-.3]); obj.reward_normal = torch.tensor([1.,0.,0.])
    obj.reward_offset = torch.tensor([.0256,0.,.1 if settings is None else 0.])
    obj.residual_raw = torch.linspace(-.05,.05,count*18).reshape(count,18)
    obj.action_delta = torch.linspace(0.,.1,count)
    obj.ball_observation_enabled = torch.ones(count,dtype=torch.bool)
    return obj, tool_v, tool_w, ball_v


def event_phase(obj, phase):
    obj.flight_qualified[0] = phase >= 1
    for name in ['legal_initial_contact','string_hit','airborne_string_hit','had_contact']:
        getattr(obj.rally,name)[:2] = phase >= 2
    if phase >= 3:
        obj.rally.released[:2] = True
        obj.rally.outgoing_velocity[:2] = torch.tensor([[6.,0.,-1.],[8.1,0.,-1.1]])
    if phase >= 4:
        obj.rally.crossed_net[:2] = True
    if phase >= 5:
        obj.rally.ground_seen[:2] = True; obj.rally.landed_target[:2] = True
        obj.rally.landing_point[:2] = torch.tensor([3.,0.,.025])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); assert not args.output.exists()
    run=args.source_root.resolve()/'reports/G07_full_racket_v4/residual_ppo_joint_placement_actor32_n2048_probe8_fixed_train128_01'
    assert json.loads((run/'job.json').read_text())['status']=='SUCCEEDED'
    with zipfile.ZipFile(run/'source_snapshot.zip') as archive: legacy=archive.read(MODULE)
    assert hashlib.sha256(legacy).hexdigest()==json.loads((run/'source_hashes.json').read_text())[MODULE]
    current=(ROOT/MODULE).read_bytes()
    old_cls,new_cls=methods(legacy.decode(),'LegacyCPUFixture'),methods(current.decode(),'CandidateCPUFixture')
    public_data=np.load(run/'public_and_actions_first2.npz')
    recordings=np.load(run/'all_first_episodes.npz')
    episodes=json.loads((run/'episodes.json').read_text())
    racket_id=json.loads((run/'geometry.json').read_text())['body_names'].index('Racket')
    pairs=[(tick,env) for tick in range(0,len(public_data['observations']),2) for env in [0,1]
        if public_data['valid_first_episode'][tick,env] and 2.2<=public_data['observations'][tick,env,775]<episodes[env]['contact_time_s']]
    public=np.stack([public_data['observations'][tick,env] for tick,env in pairs])
    poses=np.stack([recordings['link_poses'][tick//2,env] for tick,env in pairs])
    balls=np.stack([recordings['ball_position_m'][tick//2,env] for tick,env in pairs])
    root_rotation=Rotation.from_quat(poses[:,0,[4,5,6,3]]).as_matrix()
    tool_rotation=Rotation.from_quat(poses[:,racket_id,[4,5,6,3]]).as_matrix()
    count=len(pairs); assert count>=2
    court=json.loads((ROOT/'configs/court_net_v4.json').read_text())
    settings=json.loads((ROOT/'configs/g07_full_racket_residual_ppo_live_alignment_v4.json').read_text())['live_alignment_reward']
    legacy_checks=[]
    for joint in [False,True]:
        old,*_=make_fixture(old_cls,poses,balls,root_rotation,public,None,court,racket_id,joint=joint)
        disabled,*_=make_fixture(new_cls,poses,balls,root_rotation,public,None,court,racket_id,joint=joint)
        for phase in range(6):
            event_phase(old,phase);event_phase(disabled,phase)
            a,b=old._get_rewards(),disabled._get_rewards()
            assert torch.equal(a,b) and torch.equal(old.last_residual_reward,disabled.last_residual_reward)
        legacy_checks.append(dict(mode='v2' if joint else 'v1',fixture_rows_per_phase=count,
            controlled_phases=6,all_reward_values_bitwise_equal=True))
    enabled,tool_v,tool_w,ball_v=make_fixture(new_cls,poses,balls,root_rotation,public,settings,court,racket_id)
    enabled._get_rewards(); observed=enabled.live_alignment_reward.fixture_inputs
    relative=balls-poses[:,racket_id,:3]
    local=np.einsum('nji,nj->ni',tool_rotation,relative)
    normal=tool_rotation[:,:,0]
    rate=np.sum(normal*(ball_v-tool_v),1)+np.sum(np.cross(tool_w,normal)*relative,1)
    local_error=float(np.max(np.abs(observed['ball_relative_tool_m'].numpy()-local)))
    rate_error=float(np.max(np.abs(observed['plane_distance_rate_mps'].numpy()-rate)))
    assert local_error<TOLERANCES['local_position_m'] and rate_error<TOLERANCES['plane_rate_mps']
    gap=local[:,0]-.0256
    active=(gap>=0)&(gap<=.8)&(rate<-.25)&(public[:,775]>=2.2)&(public[:,775]<=3.3)
    potential=np.where(active,settings['contact_potential_weight']*np.exp(-(gap/settings['normal_distance_scale_m'])**2
        -np.sum((local[:,1:]/[.0915,.1165])**2,1)),0.)
    expected=settings['discount']*potential
    potential_error=float(np.max(abs(enabled.live_alignment_reward.last_components[:,0].numpy()-expected)))
    assert potential_error<TOLERANCES['independent_potential'] and potential.max()>.1
    shifts=np.array([[0.,0.,0.],[128.,-128.,0.],[-256.,256.,0.],[512.,-512.,0.]])[np.arange(count)%4]
    translated,*_=make_fixture(new_cls,poses,balls,root_rotation,public,settings,court,racket_id,origin=shifts)
    translated._get_rewards()
    shift_error=float((translated.last_residual_reward-enabled.last_residual_reward).abs().max())
    assert shift_error<TOLERANCES['translated_reward']
    v2,*_=make_fixture(new_cls,poses,balls,root_rotation,public,None,court,racket_id)
    v3,*_=make_fixture(new_cls,poses,balls,root_rotation,public,settings,court,racket_id)
    for obj in [v2,v3]:
        event_phase(obj,5)
        # Isolate release/target wiring from the intentionally different
        # precontact position/potential terms in all remaining fixture lanes.
        obj.rally.had_contact.fill_(True)
    before=v2._get_rewards();after=v3._get_rewards()
    assert v2.residual_rewards.paid['target'][:2].tolist()==[True,True]
    assert v3.residual_rewards.paid['target'][:2].tolist()==[False,True]
    delta=v3.live_alignment_reward.last_components.sum(1).clone();delta[0]-=20.
    torch.testing.assert_close(after-before,delta,atol=TOLERANCES['payment_difference'],rtol=0.)
    assert v3.live_alignment_reward.last_components[:,0].eq(0).all()
    v3._get_rewards();assert v3.live_alignment_reward.last_components.eq(0).all()
    v3.live_alignment_reward.seen_release.fill_(True)
    v3.live_alignment_reward.seen_frame.fill_(True)
    v3.live_alignment_reward.previous_potential.copy_(torch.linspace(.1,5.,count))
    v3.residual_rewards.paid['target'].fill_(True)
    v3.joint_placement_reward.seen_ground.fill_(True)
    ids=torch.tensor([0,1]);other=v3.live_alignment_reward.seen_release[2:].clone()
    other_potential=v3.live_alignment_reward.previous_potential[2:].clone()
    sums=v3.live_alignment_reward.job_sums.clone();v3._reset_idx(ids)
    assert torch.equal(v3.fixture_parent_reset_ids,ids)
    assert not v3.live_alignment_reward.seen_release[:2].any()
    assert not v3.live_alignment_reward.seen_frame[:2].any()
    assert v3.live_alignment_reward.previous_potential[:2].eq(0).all()
    assert torch.equal(v3.live_alignment_reward.seen_release[2:],other)
    assert torch.equal(v3.live_alignment_reward.previous_potential[2:],other_potential)
    assert not v3.residual_rewards.paid['target'][:2].any() and v3.residual_rewards.paid['target'][2:].all()
    assert not v3.joint_placement_reward.seen_ground[:2].any() and v3.joint_placement_reward.seen_ground[2:].all()
    assert torch.equal(v3.live_alignment_reward.job_sums,sums)
    report=dict(checked_at=datetime.now(timezone.utc).isoformat(),
        scope=f'CPU numerical integration fixture executing actual reward method bodies.{count} recorded precontact geometries/velocities with controlled flags, costs and a stub parent reset. No Isaac constructor, PhysX step, historical payment reconstruction or policy-performance result.',
        actual_precontact_recording_rows=count,actual_source_cases=[0,1],
        legacy_compatibility=legacy_checks,maximum_local_coordinate_error_m=local_error,
        maximum_plane_distance_rate_error_mps=rate_error,maximum_independent_potential_error=potential_error,
        maximum_translated_origin_reward_error=shift_error,original_target_gate_soft_and_attack=[True,True],
        candidate_target_gate_soft_and_attack=[False,True],new_components_not_paid_twice=True,
        selective_reset_hook_observed=True,other_lane_release_state_preserved=True,job_totals_preserved=True,
        tolerances=TOLERANCES,
        source_sha256={str(ROOT/MODULE):hashlib.sha256(current).hexdigest(),
            'actual_native8_source_snapshot_reward_module':hashlib.sha256(legacy).hexdigest(),
            str(run/'source_snapshot.zip'):sha(run/'source_snapshot.zip'),
            str(run/'public_and_actions_first2.npz'):sha(run/'public_and_actions_first2.npz'),
            str(run/'all_first_episodes.npz'):sha(run/'all_first_episodes.npz'),
            str(Path(__file__).resolve()):sha(Path(__file__).resolve())},
        limitations=['Rally flags, torque costs and parent reset in this check are controlled CPU fixtures, not recorded engine events.',
            'Other body velocities are unused zeros; only actual public root/tool/ball velocities are reconstructed.',
            'The check does not establish native hook timing, physical ground reset, actuator response or improved policy performance.',
            'Translation error includes expected float32 quantization at512m environment offsets; test tolerances do not change the original acceptance protocol.'],
        native_execution=False,formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix('.json.tmp');temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(args.output)
    print(json.dumps({key:report[key] for key in ['actual_precontact_recording_rows','legacy_compatibility',
        'maximum_local_coordinate_error_m','maximum_plane_distance_rate_error_mps',
        'maximum_independent_potential_error','maximum_translated_origin_reward_error','native_execution']}),flush=True)


if __name__=='__main__':
    main()
