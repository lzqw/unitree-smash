"""Lossless first-episode inputs and dispatched actions for fixed-policy trials.

Every selected case is retained, including failures. Outcome labels remain the
independently checked original criteria; this module does not label all recorded
actions as expert demonstrations or restore simulator state.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .episode_trace_v4 import PublicHistoryPacker, unpack_public


class FixedPolicyTraceV1:
    def __init__(self, output, count, max_steps, provenance):
        if type(count) is not int or count < 1 or type(max_steps) is not int or max_steps < 1:
            raise ValueError('Require positive fixed trial dimensions')
        self.output = Path(output)
        if self.output.exists():
            raise ValueError('Never overwrite a fixed-policy recording')
        self.count, self.max_steps = count, max_steps
        self.provenance = dict(provenance)
        self.packer = PublicHistoryPacker(count)
        self.packed = np.empty((count, max_steps, 104), dtype=np.float32)
        self.actions = np.empty((count, max_steps, 18), dtype=np.float32)
        self.lengths = np.zeros(count, dtype=np.int64)
        self.finished = np.zeros(count, dtype=bool)
        self.hashes = [hashlib.sha256() for _ in range(count)]
        self.last_clock = np.zeros(count, dtype=np.float32)
        self.output.mkdir(parents=True)

    def append(self, public, dispatched_actions, episode_ids):
        if (dispatched_actions.shape != (self.count, 18) or dispatched_actions.dtype != np.float32
                or not np.isfinite(dispatched_actions).all() or np.abs(dispatched_actions).max() > 1.000001):
            raise ValueError('Require finite original18D dispatched actions')
        episode_ids = np.asarray(episode_ids)
        if episode_ids.shape != (self.count,) or np.any(episode_ids[~self.finished] != 0):
            raise ValueError('Unclosed first episode reset unexpectedly')
        packed, _ = self.packer.pack(public, episode_ids)
        ids = np.flatnonzero(~self.finished)
        lengths = self.lengths[ids]
        if np.any(lengths >= self.max_steps):
            raise ValueError('Fixed trial exceeded its declared recording capacity')
        clock = public[ids, 775]
        expected = np.where(lengths == 0, 0., self.last_clock[ids]+.01)
        if np.any(np.abs(clock-expected) > 2e-6):
            raise ValueError('Require every10ms input from the real zero-clock first episode')
        self.packed[ids, lengths] = packed[ids]
        self.actions[ids, lengths] = dispatched_actions[ids]
        for i in ids:
            self.hashes[i].update(public[i].tobytes())
        self.lengths[ids] += 1
        self.last_clock[ids] = clock

    def close(self, done):
        done = np.asarray(done)
        if done.shape != (self.count,) or done.dtype != np.bool_:
            raise ValueError('Require one actual termination flag per lane')
        if np.any(done & ~self.finished & (self.lengths == 0)):
            raise ValueError('No executed command precedes episode termination')
        self.finished |= done

    def save(self, rows, outcomes):
        if not self.finished.all() or len(rows) != self.count or len(outcomes) != self.count:
            raise ValueError('Require all complete first episodes and independent outcomes')
        if (self.output/'manifest.json').exists():
            raise ValueError('Preserve the completed recording')
        manifest = dict(schema='fixed-policy-native-first-episode-trace-v1', provenance=self.provenance,
                        control_dt_s=.01, observation_size=794, action_size=18,
                        action_interface='absolute-joint-limits-v1',
                        all_selected_cases_retained=True, failures_are_not_full_expert_demonstrations=True,
                        formal_acceptance=False, episodes=[])
        # Validate the whole batch before publishing any per-case files.
        for i, (row, outcome) in enumerate(zip(rows, outcomes)):
            if row['env_id'] != i or row['episode_index'] != 0 or not row['ground_init']:
                raise ValueError('Only actual first ground episodes are eligible')
            if set(outcome) != {'airborne_contact', 'conditional_recovery', 'complete', 'fault'}:
                raise ValueError('Require original independently checked outcome fields')
            restored = unpack_public(self.packed[i, :self.lengths[i]])
            if hashlib.sha256(restored.tobytes()).hexdigest() != self.hashes[i].hexdigest():
                raise ValueError('Reconstructed public input differs from actual policy input')
        for i, (row, outcome) in enumerate(zip(rows, outcomes)):
            length = int(self.lengths[i])
            path = self.output/f'case_{row["source_training_index"]:03d}.npz'
            with path.open('xb') as stream:
                np.savez_compressed(stream, packed_public=self.packed[i, :length],
                                    dispatched_actions=self.actions[i, :length])
            record = dict(env_id=i, case_id=row['case_id'], source_training_index=row['source_training_index'],
                          samples=length, last_command_clock_s=float(self.last_clock[i]),
                          data_file=path.name, data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          full_public794_sha256=self.hashes[i].hexdigest(),
                          bitwise_public_reconstruction_verified=True, original_outcomes=outcome,
                          full_expert_under_original_task=bool(outcome['complete'] and row['numerical_failures'] == 0),
                          raw_episode=row)
            manifest['episodes'].append(record)
        temporary = self.output/'manifest.json.tmp'
        temporary.write_text(json.dumps(manifest, indent=2)+'\n')
        temporary.replace(self.output/'manifest.json')
        return dict(episodes=len(rows), samples=int(self.lengths.sum()),
                    original_complete_expert_episodes=sum(r['full_expert_under_original_task'] for r in manifest['episodes']),
                    manifest=self.output.name+'/manifest.json',
                    manifest_sha256=hashlib.sha256((self.output/'manifest.json').read_bytes()).hexdigest())
