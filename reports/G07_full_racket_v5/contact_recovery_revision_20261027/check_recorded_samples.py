"""Replay actual recorded native metric inputs on CPU; never claim new physics."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from quadsmash.tasks.stand.contact_flight_recovery_v5 import ContactFlightRecoveryV5, independent_recovery
from quadsmash.tasks.stand.contact_recovery_origin0_v5 import PROTOCOL as REVISION, validate_recovery_protocol


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    protocol = json.loads((ROOT / 'configs/evaluation_protocol_v2.json').read_text())
    original = validate_recovery_protocol(json.loads((ROOT / REVISION).read_text()))
    recording = ROOT.parent / 'unitree_smash/reports/G07_full_racket_v4/recovery_sequence_replay7_n128_01/recovery_sequence'
    observation = json.loads((recording / 'OBSERVATION.json').read_text())
    columns = observation['columns']
    output = []
    for episode in observation['episodes']:
        path = recording / episode['data_file']
        assert sha(path) == episode['data_sha256']
        with np.load(path) as source:
            values = torch.from_numpy(source['values'].copy())
            frames = source['actual_physics_frames'].copy()
        tracker = ContactFlightRecoveryV5(1, 'cpu', protocol)
        jump = SimpleNamespace()
        names = dict(flight_candidate='flight_candidate', flight_qualified='flight_qualified',
                     valid_jump_landed='valid_jump_landed', takeoff_time='takeoff_time_s',
                     landing_time='landing_time_s', flight_apex_rise='flight_apex_rise_m',
                     flight_clear_time='flight_clear_time_s', body_ground_seen='body_ground_seen',
                     maximum_body_ground_force='maximum_body_ground_force_n')
        first_hit = True
        for index, sample in enumerate(values):
            for name, column in names.items():
                value = sample[columns.index(column)].reshape(1)
                setattr(jump, name, value.bool() if name in ('flight_candidate', 'flight_qualified',
                        'valid_jump_landed', 'body_ground_seen') else value)
            def value(name):
                return sample[columns.index(name)]
            recovery = protocol['recovery']
            stable = ((value('landed_once') > 0) & (value('supported_feet') > 0)
                      & (torch.maximum(value('roll_rad').abs(), value('pitch_rad').abs())
                         <= recovery['maximum_abs_roll_pitch_deg'] * torch.pi / 180)
                      & (value('horizontal_speed_mps') <= recovery['maximum_horizontal_root_speed_m_s'])
                      & (value('angular_speed_rad_s') <= recovery['maximum_root_angular_speed_rad_s'])
                      & (value('maximum_joint_speed_rad_s') <= recovery['maximum_abs_joint_velocity_rad_s'])
                      & (~jump.body_ground_seen[0]))
            dt = ((frames[index] - frames[index - 1]) * original['physics_dt_s']) if index else .001
            tracker.metric(jump, stable.reshape(1), float(dt))
            if first_hit and value('airborne_string_hit') > 0:
                tracker.contact(jump, legal_air_strings_touching=torch.tensor([True]),
                    time=value('actual_contact_time_s').reshape(1),
                    rally_contact_time=value('actual_contact_time_s').reshape(1))
                first_hit = False
        record = tracker.record(jump, 0, float(value('root_z_m')))
        old = episode['closure']['actual_jump_record']
        row = dict(jump=old, contact_time_s=record['rally_contact_time_s'])
        recovered = independent_recovery(record, row, protocol, original['physics_dt_s'])
        assert recovered == (episode['env_id'] == 69)
        assert not old['qualified_jump_landed']
        output.append(dict(env_id=episode['env_id'], actual_samples=len(values),
                           input_sha256=sha(path), original_jump_record=old,
                           revised_association=record, independent_recovered=recovered))
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(),
                  actual_recorded_input_samples=sum(row['actual_samples'] for row in output), episodes=output,
                  unchanged_original_protocol_fields_validated=True,
                  revised_protocol_sha256=sha(ROOT / REVISION),
                  helper_source_sha256=sha(ROOT / 'quadsmash/tasks/stand/contact_flight_recovery_v5.py'),
                  checker_source_sha256=sha(Path(__file__)),
                  new_native_execution=False, historical_results_rewritten=False,
                  scoring_version_explicit=True, formal_acceptance=False)
    with (HERE / 'ACTUAL_RECORDED_INPUT_CPU_CHECK.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(actual_recorded_input_samples=result['actual_recorded_input_samples'],
                         recovered={row['env_id']: row['independent_recovered'] for row in output},
                         new_native_execution=False)))


if __name__ == '__main__':
    main()
