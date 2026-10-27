"""Check causal bounded commands and conservation on real public inputs."""
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
MAIN = HERE.parents[2]
ROOT = MAIN.parent / 'unitree_smash_contact_recovery'
sys.path.insert(0, str(ROOT))
from quadsmash.control.full_racket_origin0_residual_v4 import make_rally_controller
from scripts.initialize_centered_policy_v4 import same_tree


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    spec = importlib.util.spec_from_file_location('countermotion_proposal', HERE/'countermotion.py')
    factory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(factory)
    measured = HERE.parent/'measured_swing9_n128_01'
    inputs = json.loads((measured/'INPUTS.json').read_text())
    defaults, limits = [torch.tensor(inputs[k],dtype=torch.float32) for k in
                        ['default_joint_position_rad','joint_limits_rad']]
    policy = MAIN/'reports/G07_full_racket_v4/gain_body8_sequential_n128_01/best_candidate.pt'
    source = torch.load(policy,map_location='cpu',weights_only=False)
    before = copy.deepcopy(source)
    with np.load(measured/'public_and_actions_000.npz') as data:
        public, native, valid = [data[k].copy() for k in ['observations','actions','valid_first_episode']]
        assert data['training_indices'].tolist() == [0,29,69,100,112,123]
    assert public.shape == (800,6,794) and valid.all()
    # Confirm that30rad/s is the unchanged leg simulation limit; the arm
    # limit remains5rad/s and this prototype never changes arm commands.
    hardware_path = ROOT/'quadsmash/tasks/stand/coupled_env.py'
    hardware = hardware_path.read_text()
    assert 'velocity_limit_sim=30.0' in hardware and 'velocity_limit_sim=5.0' in hardware
    baseline = make_rally_controller(source,6,'cpu')
    expected = [baseline.action({'policy':torch.from_numpy(o)},defaults,limits).clone() for o in public]
    results = []
    for cap in [0.,2.,5.]:
        controller = factory.AirCountermotionController(make_rally_controller(source,6,'cpu'),
            source['asset_sha256'],6,'cpu',cap)
        max_native = max_delta = max_target_delta = max_slew = 0.
        max_request = max_conservation_error = max_predicted_velocity = 0.
        previous = None
        active_rows = 0
        errors_before, errors_after = [], []
        saved_state = []
        for values, actual_native, base in zip(public,native,expected):
            obs = torch.from_numpy(values)
            original = obs.clone()
            actual = controller.action({'policy':obs},defaults,limits)
            assert torch.equal(obs,original) and actual.shape == (6,18)
            assert torch.isfinite(actual).all() and (abs(actual)<=1).all()
            assert torch.equal(actual[:,12:],base[:,12:])
            target = limits[:,0]+.5*(actual+1)*(limits[:,1]-limits[:,0])
            base_target = limits[:,0]+.5*(base+1)*(limits[:,1]-limits[:,0])
            max_native = max(max_native,float(abs(actual.numpy()-actual_native).max()))
            max_delta = max(max_delta,float(abs(actual-base).max()))
            max_target_delta = max(max_target_delta,float(abs(target-base_target).max()))
            if previous is not None:
                max_slew = max(max_slew,float(abs(target[:,12:]-previous[:,12:]).max())/.01)
            previous = target.clone()
            if cap:
                diagnostic = controller.last_diagnostic
                active = diagnostic['active']
                assert torch.equal(actual[~active],base[~active])
                assert not np.any(active & (values[:,773]<=.5))
                height = 1.80-np.sum((-values[:,6:9])*values[:,760:763],axis=1)
                assert not np.any(active & ((values[:,57:61]>.5).any(1) | (height<.30) | (-values[:,8]<=.25)))
                active_rows += int(active.sum())
                max_request = max(max_request,float(abs(diagnostic['requests']).max()))
                for row in diagnostic['model_predictions']:
                    assert not row['infeasible_velocity_box']
                    assert row['weighted_error_after'] <= row['weighted_error_before']+1e-7
                    errors_before.append(row['weighted_error_before'])
                    errors_after.append(row['weighted_error_after'])
                    lane = row['lane']
                    q = defaults.numpy()[None]+values[lane:lane+1,9:27]
                    qd = values[lane:lane+1,27:45].astype(np.float64)
                    w = values[lane:lane+1,3:6].astype(np.float64)
                    shifted = qd.copy()
                    shifted[0,:12] += np.asarray(row['requested_leg_velocity_delta_rad_s'])
                    max_predicted_velocity = max(max_predicted_velocity,float(abs(shifted[:,:12]).max()))
                    old = controller.requester.model.forward(q,qd,root_angular=w)['angular_momentum']
                    new = controller.requester.model.forward(q,shifted,
                        root_angular=np.asarray(row['predicted_root_angular_body_rad_s'])[None])['angular_momentum']
                    max_conservation_error = max(max_conservation_error,float(abs(old-new).max()))
        assert max_slew <= 5.0005 and max_target_delta <= cap*.03+1e-5
        assert max_predicted_velocity <= 30.+1e-8 and max_conservation_error < 1e-10
        assert same_tree(source,before)
        if cap == 0.:
            assert max_delta == 0. and max_native <= 5e-4
        else:
            assert active_rows > 0 and max_delta > 0.
        # Selective reset preserves another lane's next action, including
        # the original source actor history and mixer buffers.
        twin = factory.AirCountermotionController(make_rally_controller(source,6,'cpu'),
            source['asset_sha256'],6,'cpu',cap)
        controller.reset()
        for c in [controller,twin]:
            c.action({'policy':torch.from_numpy(public[0])},defaults,limits)
        controller.reset(torch.tensor([0]))
        next_values = public[1].copy();next_values[0]=public[0,0]
        a = controller.action({'policy':torch.from_numpy(next_values)},defaults,limits)
        b = twin.action({'policy':torch.from_numpy(public[1])},defaults,limits)
        assert torch.equal(a[1:],b[1:])
        result = dict(maximum_velocity_change_rad_s=cap,actual_public_rows=4800,actual_post_hit_airborne_rows=active_rows,
            maximum_source_action_difference=max_delta,maximum_original_GPU_action_difference=max_native,
            maximum_leg_command_target_change_rad=max_target_delta,maximum_arm_target_rate_rad_s=max_slew,
            maximum_residual_request=max_request,maximum_hypothetical_total_leg_velocity_rad_s=max_predicted_velocity,
            maximum_model_angular_momentum_conservation_error=max_conservation_error,
            mean_model_angular_error_before=(float(np.mean(errors_before)) if errors_before else None),
            mean_model_angular_error_after=(float(np.mean(errors_after)) if errors_after else None),
            pre_contact_and_grounded_commands_bitwise_unchanged=True,arm_commands_bitwise_unchanged=True,
            low_or_overturned_or_foot_supported_commands_bitwise_unchanged=True,
            public_inputs_and_source_weights_unchanged=True,selective_reset_preserves_other_lanes=True,
            physical_tracking_or_recovery_improvement_proven=False)
        results.append(result)
        print(json.dumps(result),flush=True)
    files = [Path(__file__),HERE/'countermotion.py',policy,hardware_path,measured/'INPUTS.json',
        measured/'public_and_actions_000.npz',factory.MODEL/'known_model.py',factory.MODEL/'MODEL.json',
        factory.MODEL/'ACTUAL_MODEL_CHECK.json']
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(),results=results,
        unchanged_hardware_leg_velocity_limit_rad_s=30.,unchanged_hardware_arm_velocity_limit_rad_s=5.,
        original_residual_position_scale_rad=.15,position_preview_horizon_s=.03,
        controller_activation_minimum_public_root_height_m=.30,
        controller_activation_minimum_public_body_up_z=.25,
        known_admitted_command_height_m=1.80,
        source_sha256={str(p):sha(p) for p in files},native_execution=False,
        formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    with (HERE/'CPU_CONTROLLER_CHECK.json').open('x') as stream:
        stream.write(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
