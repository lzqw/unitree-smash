"""Associate recovery with the flight containing the actual airborne string hit."""
import math

import torch


SCHEMA = 'contact-flight-recovery-v1'


class ContactFlightRecoveryV5:
    def __init__(self, count, device, protocol):
        self.protocol = protocol
        for name in ('flight_id', 'contact_flight_id'):
            setattr(self, name, torch.zeros(count, dtype=torch.long, device=device))
        for name in ('bound', 'qualified_landing', 'stable_seen'):
            setattr(self, name, torch.zeros(count, dtype=torch.bool, device=device))
        for name in ('last_takeoff', 'rally_contact_time', 'string_time', 'takeoff_time',
                     'landing_time', 'rise', 'clear_time', 'stable_run', 'longest_stable'):
            setattr(self, name, torch.zeros(count, device=device))

    def reset(self, ids):
        for value in vars(self).values():
            if isinstance(value, torch.Tensor):
                value[ids] = 0

    @torch.inference_mode()
    def metric(self, jump, stable, dt):
        takeoff = jump.flight_candidate & (jump.takeoff_time != self.last_takeoff)
        self.flight_id += takeoff.long()
        self.last_takeoff.copy_(jump.takeoff_time)
        same = self.bound & (self.contact_flight_id == self.flight_id)
        self.rise.copy_(torch.where(same, torch.maximum(self.rise, jump.flight_apex_rise), self.rise))
        self.clear_time.copy_(torch.where(same, torch.maximum(self.clear_time, jump.flight_clear_time), self.clear_time))
        landed = (same & (~jump.flight_candidate) & jump.valid_jump_landed & jump.flight_qualified
                  & (jump.landing_time >= self.string_time))
        first = landed & (~self.qualified_landing)
        self.landing_time[first] = jump.landing_time[first]
        self.qualified_landing |= landed
        # A later unsupported interval restarts the continuous stable window;
        # it cannot erase the already observed landing of the contact flight.
        eligible = self.qualified_landing & stable & (~jump.body_ground_seen)
        self.stable_run.copy_(torch.where(eligible, self.stable_run + dt, 0.))
        torch.maximum(self.longest_stable, self.stable_run, out=self.longest_stable)
        self.stable_seen |= self.stable_run >= self.protocol['recovery']['continuous_stable_window_s'] - 1e-6

    @torch.inference_mode()
    def contact(self, jump, *, legal_air_strings_touching, time, rally_contact_time):
        bind = (legal_air_strings_touching & (~self.bound) & jump.flight_candidate
                & (self.flight_id > 0))
        self.bound |= bind
        self.contact_flight_id[bind] = self.flight_id[bind]
        self.rally_contact_time[bind] = rally_contact_time[bind]
        self.string_time[bind] = time[bind]
        self.takeoff_time[bind] = jump.takeoff_time[bind]
        self.rise[bind] = jump.flight_apex_rise[bind]
        self.clear_time[bind] = jump.flight_clear_time[bind]

    def success(self, jump):
        return self.bound & self.qualified_landing & self.stable_seen & (~jump.body_ground_seen)

    def record(self, jump, index, root_height):
        return dict(schema=SCHEMA, bound=bool(self.bound[index]),
                    contact_flight_id=int(self.contact_flight_id[index]),
                    rally_contact_time_s=float(self.rally_contact_time[index]),
                    airborne_string_time_s=float(self.string_time[index]),
                    contact_flight_takeoff_s=float(self.takeoff_time[index]),
                    qualified_contact_landing=bool(self.qualified_landing[index]),
                    contact_flight_landing_s=float(self.landing_time[index]),
                    contact_flight_com_rise_m=float(self.rise[index]),
                    contact_flight_clear_s=float(self.clear_time[index]),
                    stable_recovery=bool(self.stable_seen[index]),
                    recovery_s=float(self.longest_stable[index]),
                    body_landing=bool(jump.body_ground_seen[index]),
                    maximum_body_ground_force_n=float(jump.maximum_body_ground_force[index]),
                    terminal_root_height_m=float(root_height),
                    success=bool(self.success(jump)[index] and root_height >= .12))


def independent_recovery(record, row, original_protocol, physics_dt):
    """Check this version's raw association and unchanged physical thresholds."""
    if record.get('schema') != SCHEMA:
        raise ValueError('Missing explicit contact-flight recovery schema')
    scalar_names = ('rally_contact_time_s', 'airborne_string_time_s', 'contact_flight_takeoff_s',
                    'contact_flight_landing_s', 'contact_flight_com_rise_m', 'contact_flight_clear_s',
                    'recovery_s', 'maximum_body_ground_force_n', 'terminal_root_height_m')
    if not all(math.isfinite(record[name]) for name in scalar_names):
        raise ValueError('Nonfinite contact-flight recovery evidence')
    recovery = original_protocol['recovery']
    jump = original_protocol['jump']
    associated = (record['bound'] and record['contact_flight_id'] > 0
                  and abs(record['rally_contact_time_s'] - row['contact_time_s']) <= physics_dt
                  and record['contact_flight_takeoff_s'] <= record['airborne_string_time_s']
                  and record['airborne_string_time_s'] >= row['contact_time_s'] - physics_dt
                  and record['contact_flight_landing_s'] >= record['airborne_string_time_s']
                  and record['contact_flight_com_rise_m'] >= jump['minimum_total_com_rise_from_takeoff_m']
                  and record['contact_flight_clear_s'] >= jump['minimum_continuous_all_feet_airborne_s'] - 1e-7)
    return bool(associated and record['qualified_contact_landing'] and record['stable_recovery']
                and record['recovery_s'] >= recovery['continuous_stable_window_s'] - 1e-6
                and not record['body_landing'] and not row['jump']['body_landing']
                and not row['jump']['terminated']
                and record['maximum_body_ground_force_n'] <= recovery['maximum_body_ground_force_n']
                and row['jump']['maximum_body_ground_force_n'] <= recovery['maximum_body_ground_force_n']
                and record['terminal_root_height_m'] >= .12)
