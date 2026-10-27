"""Read existing jump/rally state at metric samples; never return actions or rewards."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


COLUMNS = (
    'episode_index', 'jump_clock_s', 'control_clock_s', 'supported_feet',
    'flight_candidate', 'flight_qualified', 'valid_jump_landed', 'airborne_seen',
    'landed_once', 'recovery_seen', 'recovery_run_s', 'longest_recovery_s',
    'body_ground_seen', 'maximum_body_ground_force_n', 'rewarded_jump',
    'takeoff_time_s', 'landing_time_s', 'flight_apex_rise_m', 'flight_clear_time_s',
    'launched', 'rally_in_play', 'had_contact', 'legal_initial_contact',
    'string_hit', 'airborne_string_hit', 'actual_contact_time_s',
    'released', 'crossed_net', 'ground_seen', 'landed_target',
    'root_z_m', 'roll_rad', 'pitch_rad', 'horizontal_speed_mps',
    'angular_speed_rad_s', 'maximum_joint_speed_rad_s',
)


class RecoverySequenceRecorder:
    def __init__(self, env, output, lanes=(69, 112), *, maximum_samples=147456,
                 native_source_confirmed=False):
        self.output = Path(output)
        if self.output.exists():
            raise ValueError('Preserve earlier observations; use a fresh output')
        self.output.mkdir(parents=True)
        self.lanes = tuple(lanes)
        if len(set(self.lanes)) != len(self.lanes) or any(i < 0 or i >= env.num_envs for i in self.lanes):
            raise ValueError('Require distinct actual environment indices')
        self.ids = torch.tensor(self.lanes, dtype=torch.long, device=env.device)
        self.buffer = torch.empty((maximum_samples, len(self.lanes), len(COLUMNS)),
                                  dtype=torch.float32, device=env.device)
        self.frames = np.empty(maximum_samples, dtype=np.int64)
        self.cursor = 0
        self.last_frame = -1
        self.first_closed = {}
        self.native_source_confirmed = bool(native_source_confirmed)

    @torch.inference_mode()
    def on_physics(self, env, roll_pitch):
        frame = int(env._jump_physics_frame)
        if frame <= self.last_frame:
            raise ValueError('Expected distinct chronological production metric samples')
        if self.cursor >= len(self.buffer):
            raise ValueError('Bounded observation buffer exhausted; preserve the job failure')
        self.last_frame = frame
        d, rally = env.robot.data, env.rally
        values = (
            env.episode_index, env.jump_clock, env.episode_length_buf * env.step_dt,
            env.observed_foot_contact.sum(1), env.flight_candidate, env.flight_qualified,
            env.valid_jump_landed, env.airborne_seen, env.landed_once, env.recovery_seen,
            env.recovery_run, env.recovery_time, env.body_ground_seen,
            env.maximum_body_ground_force, env.rewarded_jump, env.takeoff_time,
            env.landing_time, env.flight_apex_rise, env.flight_clear_time,
            env.launch_done, rally.in_play, rally.had_contact, rally.legal_initial_contact,
            rally.string_hit, rally.airborne_string_hit, rally.contact_time,
            rally.released, rally.crossed_net, rally.ground_seen, rally.landed_target,
            d.root_pos_w[:, 2], roll_pitch[:, 0], roll_pitch[:, 1],
            d.root_link_lin_vel_w[:, :2].norm(dim=1), d.root_ang_vel_w.norm(dim=1),
            d.joint_vel.abs().amax(1),
        )
        assert len(values) == len(COLUMNS)
        # Only this recorder's storage is written. No .item(), CPU copy, RNG
        # call, controller invocation, physics step or reward calculation here.
        selected = torch.stack([v[self.ids].to(torch.float32) for v in values], dim=1)
        self.buffer[self.cursor].copy_(selected)
        self.frames[self.cursor] = frame
        self.cursor += 1

    def on_reward(self, _env, _features, _reward):
        # G07 overrides the legacy reward and does not call this hook.
        return None

    def on_episodes(self, env, ids):
        # FullRacketEnv increments episode_index and clears rally state before
        # JumpEnv invokes this hook. Read the just-appended jump record; do not
        # interpret the already-reset rally flags as terminal observations.
        latest = {row['env_id']: row for row in env.episode_records[-len(ids):]}
        for lane in ids.detach().cpu().tolist():
            if lane not in self.lanes or lane in self.first_closed:
                continue
            if int(env.episode_index[lane]) != 1:
                raise ValueError('Observer was not attached before the first ground episode')
            self.first_closed[lane] = dict(actual_closed_episode_index=0,
                native_episode_index_at_post_rally_reset_hook=1,
                terminal_control_clock_s=float(env.episode_length_buf[lane] * env.step_dt),
                actual_jump_record=dict(latest[lane]))

    def finish(self, *, require_complete=True):
        if require_complete and set(self.first_closed) != set(self.lanes):
            raise ValueError('Missing one of the selected first-episode closures')
        data = self.buffer[:self.cursor].detach().cpu().numpy().copy()
        frames = self.frames[:self.cursor].copy()
        records = []
        for j, lane in enumerate(self.lanes):
            keep = data[:, j, COLUMNS.index('episode_index')] == 0
            values = data[keep, j]
            actual_frames = frames[keep]
            if not len(values) or (np.diff(actual_frames) <= 0).any():
                raise ValueError('Missing or unordered actual metric samples')
            finite = np.delete(values, COLUMNS.index('actual_contact_time_s'), axis=1)
            if not np.isfinite(finite).all():
                raise ValueError('Nonfinite state in actual metric observation')
            # Contact time is NaN until an actual contact, matching production.
            contacts = values[:, COLUMNS.index('actual_contact_time_s')]
            if np.isinf(contacts).any():
                raise ValueError('Infinite contact event timestamp')
            path = self.output / f'lane{lane:04d}_first_episode_metrics.npz'
            if path.exists():
                raise ValueError('Preserve an earlier metric trace')
            with path.open('xb') as stream:
                np.savez_compressed(stream, values=values, actual_physics_frames=actual_frames,
                                    columns=np.asarray(COLUMNS))
            records.append(dict(env_id=lane, original_episode_index=0, samples=len(values),
                data_file=path.name, data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                closure=self.first_closed.get(lane),
                scope='Existing production metric sampling times; no interpolation or new criteria.'))
        result = dict(created_at=datetime.now(timezone.utc).isoformat(), columns=COLUMNS,
            episodes=records, total_observer_callbacks=self.cursor,
            native_source_confirmed=self.native_source_confirmed,
            new_physics_or_rewards_supplied=False, scoring_changed=False,
            formal_cases_read=False, formal_acceptance=False,
            scope='Read-only jump/rally state at original metric samples. Recording alone does not prove same-execution reproduction, correct recovery qualification, or task success.')
        with (self.output / 'OBSERVATION.json').open('x') as stream:
            stream.write(json.dumps(result, indent=2) + '\n')
        return result
