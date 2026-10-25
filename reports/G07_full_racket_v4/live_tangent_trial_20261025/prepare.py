"""Create unexecuted bounded public-feedback policies and check real inputs."""
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(HERE)]
from reactive import PublicTangentCorrection, ResidualWithTangent
from quadsmash.control.full_racket_kinematics_v4 import FullRacketKinematicsV4
from quadsmash.control.full_racket_residual_v4 import FullRacketResidualControllerV4

SOURCE = ROOT / 'reports/G07_full_racket_v4/residual_ppo_joint_placement_actor32_n2048_probe8_01/policy.pt'
RUN = ROOT / 'reports/G07_full_racket_v4/residual_ppo_joint_placement_actor32_n2048_probe8_fixed_train128_01'
GAINS = [('gain035', .35), ('gain070', .70)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (HERE/'PREPARATION.json').exists(), 'Preserve existing preparation'
    assert not any((HERE/name).exists() for name, _ in GAINS)
    assert sha(SOURCE) == '846656e9d02c74f4b72929253ed9196557af4f9affe39236fc55d7f4bc5d744c'
    assert json.loads((RUN/'completed.json').read_text())['env_steps'] == 102400
    source = torch.load(SOURCE, map_location='cpu', weights_only=False)
    assert source['native_execution'] and not source['formal_acceptance']
    actual = np.load(RUN/'public_and_actions_first2.npz')
    public = torch.tensor(actual['observations'].reshape(-1, 794))
    assert public.shape == (1600, 794) and torch.isfinite(public).all()
    defaults = json.loads((ROOT/'configs/full_racket_reset_v4.json').read_text())['arm_joint_positions_rad']
    events = json.loads((RUN/'contacts.json').read_text())
    q = torch.tensor([e['joint_position_rad'][12:] for e in events], dtype=torch.float32)
    kin = FullRacketKinematicsV4().float()
    with torch.inference_mode():
        center, normal, _, _, orientation, _, _ = kin(q, return_pose=True)
    variants, receipts = [], []
    for name, gain in GAINS:
        actor = torch.jit.load(io.BytesIO(source['residual_actor_jit']), map_location='cpu').eval()
        correction = PublicTangentCorrection(defaults, gain).eval()
        wrapper = ResidualWithTangent(actor, correction, source['residual_joint_scale_rad']).eval()
        scripted = torch.jit.script(wrapper)
        with torch.inference_mode():
            original_public = public.clone()
            eager = wrapper(public)
            exported = scripted(public)
            split = torch.cat([scripted(public[:711]), scripted(public[711:])])
            old_actions = actor(public)
            delta, tau, gate, _, _, _ = correction.diagnostics(public)
            assert torch.equal(public, original_public)
            assert torch.isfinite(exported).all() and exported[:, 12:].abs().max() <= 1
            torch.testing.assert_close(exported, eager, atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(split, exported, atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(exported[:, :12], old_actions[:, :12], atol=2e-7, rtol=2e-6)
            assert delta.abs().max() <= .100001
            unused_changed = public.clone()
            unused_changed[:, 69:759] = 123.
            unused_changed[:, 759] = -456.
            unused_changed[:, 776:794] = -42.
            assert torch.equal(correction(unused_changed), delta)
            for column in [773, 774]:
                disabled = public.clone(); disabled[:, column] = 1.
                assert correction(disabled).eq(0).all()
            disabled = public.clone(); disabled[:, 67] = 0.
            assert correction(disabled).eq(0).all()
            # Controlled geometry fixture uses every actual measured arm
            # configuration. It is expressly not a new physical execution.
            fixture = torch.zeros(len(q), 794)
            fixture[:, 21:27] = q-torch.tensor(defaults)
            fixture[:, 45:48] = center; fixture[:, 48:51] = normal
            fixture[:, 61:64] = center+.3756*normal+.05*orientation[:, :, 2]
            fixture[:, 64:67] = -6*normal
            fixture[:, 67:69] = 1.; fixture[:, 772] = 1.; fixture[:, 775] = 2.6
            fixture[:, 763] = 1.
            fixture[:, 760:763] = torch.tensor([1.1, 0., 1.8])
            d, t, enabled, _, _, _ = correction.diagnostics(fixture)
            assert enabled.sum() >= 60
            torch.testing.assert_close(t, torch.full_like(t, .35/6), atol=2e-6, rtol=2e-6)
            new_center, _, _, _ = kin(q+d)
            desired = center+.05*orientation[:, :, 2]
            before = torch.linalg.vector_norm(desired-center, dim=1)
            after = torch.linalg.vector_norm(desired-new_center, dim=1)
            assert after[enabled].median() < before[enabled].median()
        stream = io.BytesIO(); torch.jit.save(scripted, stream); blob = stream.getvalue()
        proposal = copy.deepcopy(source)
        history = {}
        for key in ['result', 'training_result', 'initialization_training_result', 'residual_training_metadata',
                'selected_source_run', 'selected_generation', 'selected_candidate', 'selected_training_case_index']:
            if key in proposal:
                history[key] = proposal.pop(key)
        proposal.update(native_execution=False, performance_inherited=False, formal_acceptance=False,
            method='Transferred joint8 residual actor plus bounded causal public tangent interception proposal',
            kind='Unexecuted algorithm proposal; original full-racket physical and scoring contract unchanged',
            source_policy_sha256=sha(SOURCE), source_policy_history_not_inherited=history,
            residual_actor_jit=blob,
            reactive_trial=dict(schema='causal-public-tangent-proposal-v1', gain=gain,
                maximum_joint_delta_rad=.1, horizon_s=.2, damping=.04,
                original_arm_target_slew_limit_rad_s=5., source_policy=str(SOURCE.relative_to(ROOT)),
                source_actor_jit_sha256=hashlib.sha256(source['residual_actor_jit']).hexdigest(),
                actor_jit_sha256=hashlib.sha256(blob).hexdigest(),
                source_program_sha256=sha(HERE/'reactive.py'),
                no_simulator_state_writes=True, no_case_id_or_future_labels=True,
                native_execution=False, formal_acceptance=False))
        directory = HERE/name; directory.mkdir()
        torch.save(proposal, directory/'proposal.pt')
        variants.append(proposal)
        receipts.append(dict(name=name, gain=gain, proposal=str((directory/'proposal.pt').relative_to(ROOT)),
            proposal_sha256=sha(directory/'proposal.pt'), actor_jit_sha256=hashlib.sha256(blob).hexdigest(),
            actual_public_rows=len(public), enabled_native_input_rows=int(gate.sum()),
            actual_public_correction_max_rad=float(delta.abs().max()),
            native_input_enabled_tau_range_s=[float(tau[gate].min()), float(tau[gate].max())] if gate.any() else None,
            eager_script_max_error=float((eager-exported).abs().max()),
            split_batch_max_error=float((split-exported).abs().max()),
            leg_actor_max_error=float((exported[:, :12]-old_actions[:, :12]).abs().max()),
            fixture_actual_arm_configurations=len(q), fixture_enabled=int(enabled.sum()),
            fixture_static_position_error_median_before_m=float(before[enabled].median()),
            fixture_static_position_error_median_after_m=float(after[enabled].median()),
            fixture_is_native_execution=False))
    # Replay the actual controller/mixer on the saved 100Hz public sequence.
    contract = json.loads((ROOT/'reports/G07_full_racket_v4/CPU_REPLAY_AUTHORED_JOINTS_0724.json').read_text())
    names = [f'{leg}_{joint}_joint' for joint in ['hip', 'thigh', 'calf'] for leg in ['FL', 'FR', 'RL', 'RR']]+[f'joint{i}' for i in range(1, 7)]
    declared = dict(contract['defaults_before_neutral_arm_override'])
    declared.update({f'joint{i+1}':v for i,v in enumerate(defaults)})
    default = torch.tensor([declared[n] for n in names], dtype=torch.float32)
    limits = torch.tensor([contract['limits'][n] for n in names], dtype=torch.float32)
    controllers = [FullRacketResidualControllerV4(p, 2, 'cpu') for p in [source, *variants]]
    outputs = [[] for _ in controllers]
    with torch.inference_mode():
        for observations in actual['observations']:
            obs = torch.tensor(observations)
            before = obs.clone()
            for controller, stream in zip(controllers, outputs):
                result = controller.action({'policy':obs}, default, limits)
                assert torch.isfinite(result).all() and result.abs().max() <= 1.
                assert torch.equal(obs, before)
                stream.append(result.numpy().copy())
    outputs = [np.stack(stream) for stream in outputs]
    valid = actual['valid_first_episode']
    for receipt, actions in zip(receipts, outputs[1:]):
        assert np.array_equal(actions[:, :, :12], outputs[0][:, :, :12])
        targets = limits[:, 0].numpy()+.5*(actions+1)*(limits[:, 1]-limits[:, 0]).numpy()
        same_episode = valid[1:] & valid[:-1]
        slew = np.abs(np.diff(targets[:, :, 12:], axis=0))[same_episode]
        assert slew.max() <= .05001
        receipt.update(full_controller_replayed_ticks=len(actions), full_controller_leg_commands_bitwise_equal=True,
            maximum_first_episode_arm_target_step_rad=float(slew.max()),
            changed_first_episode_command_rows=int(np.any(actions!=outputs[0],axis=2)[valid].sum()))
    report = dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='CPU export, bounded geometry fixtures and exact saved-input native public-sequence replay only. No proposed controller has run in PhysX; no native performance inherited.',
        source_policy_sha256=sha(SOURCE), source_public_sha256=sha(RUN/'public_and_actions_first2.npz'),
        variants=receipts, original_physics_rules_cases_thresholds_changed=False,
        history_previous_action_and_future_goal_time_do_not_affect_correction=True,
        ball_absent_air_hit_and_landed_gates_disable_correction=True,
        actual_native_baseline_actions_CPU_max_error_on_valid_rows=float(np.abs(outputs[0]-actual['actions'])[valid].max()),
        limitations=['Short linear prediction omits future relative acceleration and normal curvature. Recompute from actual observations each 10ms.',
            'A locally improving static position fixture does not establish actuator tracking, air contact, attack, first landing, legality or recovery.',
            'Changing arm commands changes the coupled base dynamics. Frozen-input replay cannot predict those changes.',
            'The additional own-side string-center gate is conservative software protection, not a replacement for original full-volume rule checks.'],
        next_native_execution='Only after the current joint40 learning AND its frozen TRAIN128 evaluation actually finish, and only if original formal readiness is still false. Probe these two predeclared candidates on all128 original TRAIN cases under the existing GPU0 guard and unchanged protocol.',
        source_program_sha256={p.name:sha(p) for p in [Path(__file__).resolve(), HERE/'reactive.py']},
        native_execution=False, formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    (HERE/'PREPARATION.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(variants=receipts, native_execution=False), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
