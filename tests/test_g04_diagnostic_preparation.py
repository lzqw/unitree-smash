import csv
import json
import subprocess
import sys
from types import SimpleNamespace
import torch
from quadsmash.tasks.stand.jump_diagnostics import JumpDiagnosticRecorder
from quadsmash.tasks.stand.jump_reward_audit import v2c_components
from scripts.audit_g04_rewards_cpu import PROTOCOL


def test_native_diagnostic_default_and_guard_do_not_import_isaac(tmp_path):
    command = [sys.executable, 'scripts/diagnose_g04_rewards.py']
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    assert plan['maximum_interactions'] == 51200 and not plan['weights_updated']
    custom = subprocess.run([*command, '--comparison-checkpoint', str(tmp_path/'pilot50.pt')],
                            capture_output=True, text=True, check=True)
    assert json.loads(custom.stdout)['comparison_checkpoint'] == str(tmp_path/'pilot50.pt')
    result = subprocess.run([*command, '--run-native', '--output', str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 2 and 'live run_gpu_job.py parent' in result.stderr
    assert 'ModuleNotFoundError' not in result.stderr


def test_recorder_captures_pre_reset_actions_rewards_and_only_first_episodes(tmp_path):
    n = 2; zeros = torch.zeros(n); flags = torch.zeros(n, dtype=torch.bool)
    env = SimpleNamespace(num_envs=n, device='cpu', step_dt=.01, _phase=lambda: torch.ones(n),
        com_position=torch.ones(n, 3) * .4, com_velocity=torch.zeros(n, 3), reset_com_height=torch.ones(n) * .4,
        takeoff_time=zeros.clone(), landing_time=zeros.clone(), flight_candidate=flags.clone(),
        flight_qualified=flags.clone(), valid_jump_landed=flags.clone(), body_ground_seen=flags.clone(),
        observed_foot_contact=torch.ones(n, 4, dtype=torch.bool), recovery_run=zeros.clone(),
        action_before_clip=torch.ones(n, 18) * 1.2, last_action=torch.ones(n, 18),
        targets=torch.ones(n, 18) * .6, joint_ids=list(range(18)), episode_records=[])
    f = {name: zeros.clone() for name in ('height_tracking', 'flight', 'recovery', 'prepare_quality',
        'recovery_quality', 'arm_error', 'torque_square_sum', 'action_delta', 'settling', 'settle_cost')}
    f.update({name: flags.clone() for name in ('first_jump', 'first_success', 'first_body', 'terminated')})
    f['height_tracking'][:] = .5; f['target_com'] = torch.ones(n) * .45
    recorder = JumpDiagnosticRecorder(env, tmp_path, ['zero', 'policy'], PROTOCOL)
    recorder.mean[:] = 1.5; recorder.model_std[:] = .7; recorder.std[:] = .15
    reward = sum(v2c_components(f, .01).values())
    before = env.action_before_clip.clone(); recorder.on_reward(env, f, reward)
    assert torch.equal(before, env.action_before_clip)
    env.episode_records = [dict(env_id=i, success=False, body_landing=False) for i in range(n)]
    recorder.on_episodes(env, torch.arange(n))
    env.com_position.zero_(); env.action_before_clip.zero_(); recorder.on_reward(env, f, reward)
    result = recorder.finish()
    rows = list(csv.DictReader((tmp_path/'reward_trace.csv').open()))
    assert len(rows) == 2 and float(rows[0]['com_height_m']) > .39
    assert float(rows[0]['actual_action_clip_fraction']) == 1.
    assert json.loads(rows[1]['policy_std'])[0] > .69
    assert json.loads(rows[1]['execution_noise_std'])[0] < .16
    assert json.loads(rows[0]['applied_action']) == [1.] * 18
    assert result['groups']['policy']['recovery_given_qualified_landing'] is None
    assert result['reward_reconstruction_max_error'] == 0.
