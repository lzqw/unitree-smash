"""Clarify native contact log phase using saved, separated public states.

CPU analysis only. The material point is selected retrospectively from an
actual contact and is never provided to a controller. A 100 Hz sample is not
an immediately pre-impact state. All native scores and raw records stay intact.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = HERE.parents[2].parent/'unitree_smash_contact_recovery'
MODEL = BASE/'centroidal_recovery_model_20261027'
sys.path.insert(0, str(ROOT))
from quadsmash.physics.full_racket_v4 import load_racket, string_prism_gap
from quadsmash.physics.native import physics_contract


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location('verified_phase_model', MODEL/'known_model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    proof = read(MODEL/'ACTUAL_MODEL_CHECK.json')
    for name in ['known_model.py', 'MODEL.json']:
        assert sha(MODEL/name) == proof['source_sha256'][str(MODEL/name)]
    model = module.KnownModel()
    assert all(sha(Path(p)) == h for p, h in model.record['source_sha256'].items())
    racket = model.indices['Racket']
    config = load_racket()
    radius = physics_contract()['collision_radius_m']
    sources = [Path(__file__), MODEL/'known_model.py', MODEL/'MODEL.json', MODEL/'ACTUAL_MODEL_CHECK.json',
        ROOT/'quadsmash/physics/full_racket_v4.py', ROOT/'quadsmash/physics/native.py',
        ROOT/'configs/full_racket_v4_r3.json', ROOT/'quadsmash/tasks/stand/ball_env.py',
        ROOT/'quadsmash/tasks/stand/full_racket_env_v4.py', ROOT/'quadsmash/tasks/stand/observation_contract.py']
    ball_code = (ROOT/'quadsmash/tasks/stand/ball_env.py').read_text()
    event_code = (ROOT/'quadsmash/tasks/stand/full_racket_env_v4.py').read_text()
    assert 'Read the previous completed substep' in ball_code
    assert 'self._consume_physics_frame(frame)' in ball_code
    assert 'super()._consume_physics_frame(frame)' in event_code
    assert 'point_velocity_mps=contact_velocity[i].cpu().tolist()' in event_code
    results = []
    # Frozen, already-completed data only. No candidate is selected by outcome.
    for run_name, candidate in [('early_lateral7_best_cold128_01', 1),
                                 ('windowed_swing7_n128_01', 0),
                                 ('windowed_swing7_n128_01', 1)]:
        run = BASE/run_name
        suffix = f'{candidate:03d}'
        files = [run/'INPUTS.json', run/f'episodes_{suffix}.json', run/f'contacts_{suffix}.json',
                 run/f'public_and_actions_{suffix}.npz', run/'geometry.json']
        before = {str(p): sha(p) for p in files}
        sources.extend(files)
        inputs, rows, contacts = [read(p) for p in files[:3]]
        assert len(rows) == 128 and inputs['training_indices'] == list(range(128))
        assert [r['source_training_index'] for r in rows] == list(range(128))
        assert inputs['origin_spacing_m'] == 0
        assert read(run/'geometry.json')['body_names'] == model.names
        data = np.load(files[3])
        defaults = np.asarray(inputs['default_joint_position_rad'])
        actual = 0
        for column, index in enumerate(data['training_indices']):
            row = rows[int(index)]
            if not (row['legal_initial_contact'] and row['string_hit'] and row['airborne_string_hit']):
                continue
            actual += 1
            matches = [e for e in contacts if e['env_id'] == row['env_id']
                and e['episode_index'] == row['episode_index'] and e['first_racket_contact'] and e['string_contact']]
            assert len(matches) == 1
            event = matches[0]
            pose = np.asarray(event['body_link_poses_local'])[racket]
            point_local = module.rotation(pose[3:]).T@(np.asarray(event['point_m'])-pose[:3])
            obs = data['observations'][:, column].astype(np.float64)
            state = model.public_world_state(obs, defaults)
            root_R = state['root_rotation_world']
            tool_R = root_R@state['rotation'][:, racket]
            tool_position = state['body_position_world'][:, racket]
            tool_q = Rotation.from_matrix(tool_R).as_quat()[:, [3, 0, 1, 2]]
            ball_position = state['root_position_world']+np.einsum('nij,nj->ni', root_R, obs[:, 61:64])
            gap = string_prism_gap(torch.as_tensor(ball_position), torch.as_tensor(tool_position),
                torch.as_tensor(tool_q), config, radius).numpy()
            valid = (data['valid_first_episode'][:, column].astype(bool) & (obs[:, 775] < event['time_s'])
                & (obs[:, 773] < .5) & (obs[:, 67] > .5) & (gap > .001))
            eligible = np.flatnonzero(valid)
            assert len(eligible), (run_name, candidate, int(index))
            tick = int(eligible[-1])
            off = state['rotation'][tick, racket]@point_local
            point_velocity = root_R[tick]@(state['linear'][tick, racket]
                + np.cross(state['angular'][tick, racket], off))
            # Independent reconstruction of the tool origin velocity/omega
            # from their public relative fields checks the velocity convention.
            public_tool_v = obs[tick, 51:54]+obs[tick, :3]+np.cross(obs[tick, 3:6], obs[tick, 45:48])
            public_tool_w = obs[tick, 54:57]+obs[tick, 3:6]
            direct = root_R[tick]@(public_tool_v+np.cross(public_tool_w, off))
            velocity_error = float(np.linalg.norm(point_velocity-direct))
            assert velocity_error < 1e-4
            ball_velocity = root_R[tick]@(obs[tick, 64:67]+obs[tick, :3]
                + np.cross(obs[tick, 3:6], obs[tick, 61:64]))
            dt = float(event['time_s']-obs[tick, 775])
            assert 0 < dt < .03
            results.append(dict(run=run_name, candidate_id=candidate, TRAIN_index=int(index),
                saved_control_tick=tick, separated_sample_clock_s=float(obs[tick, 775]),
                contact_log_clock_s=event['time_s'], sample_before_contact_ms=1000*dt,
                sampled_sphere_string_surface_gap_m=float(gap[tick]),
                offline_material_point_local_m=point_local.tolist(),
                sampled_separated_point_velocity_world_m_s=point_velocity.tolist(),
                contact_force_substep_logged_point_velocity_world_m_s=event['point_velocity_mps'],
                sampled_separated_ball_velocity_world_m_s=ball_velocity.tolist(),
                contact_force_substep_logged_ball_velocity_world_m_s=event['ball_velocity_mps'],
                independent_public_velocity_reconstruction_error_m_s=velocity_error,
                exact_immediately_preimpact_velocity_available=False))
        assert actual == 4
        assert all(sha(Path(p)) == h for p, h in before.items())
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), results=results,
        saved_data_scope='All legal airborne first string contacts among the six pre-existing recorded public streams in each of three fixed runs. All128 native performance cases retained.',
        phase='point_velocity_mps is read from the completed PhysX contact-force substep, after sim.step; it is not an immediately-before-impact measurement.',
        sample_limit='Separated public samples are100Hz observations, not exact pre-impact states. Between-sample motion and contact recoil cannot be separated causally from these records.',
        no_new_native_run=True, no_controller_input_added=True, native_scores_changed=False,
        formal_cases_read=False, formal_acceptance=False, goal_complete=False,
        source_sha256={str(p): sha(p) for p in sorted(set(sources))})
    with (HERE/'SAVED_SEPARATED_STATE_CHECK.json').open('x') as f:
        f.write(json.dumps(result, indent=2)+'\n')
    for row in results:
        print(json.dumps({k: row[k] for k in ['run', 'candidate_id', 'TRAIN_index',
            'sample_before_contact_ms', 'sampled_sphere_string_surface_gap_m',
            'sampled_separated_point_velocity_world_m_s', 'contact_force_substep_logged_point_velocity_world_m_s']}))


if __name__ == '__main__':
    main()
