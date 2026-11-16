"""Reset-safe, long-running physical replay with a chronological ring buffer."""
import torch
from .supermimic_physical_dataset_v1 import encode_physical, valid_windows
from .supermimic_physical_world_v2 import sample_event_windows
from .supermimic_physical_capture_v1 import tensor_tree_copy


def row_tensors(row, schema):
    def memory(record):
        return dict(rally=record['task'], recovery=record['recovery'],
                    auxiliary=record['auxiliary'], paid_events=record['event_state'])
    b, a = row['before'], row['after']
    bv, bm = schema.encode(memory(b)); av, am = schema.encode(memory(a))
    return dict(state=encode_physical(b['physical']), next_state=encode_physical(a['physical']),
        values=bv, valid=bm, next_values=av, next_valid=am,
        event_state=b['event_state'], next_event_state=a['event_state'],
        targets=row['final_joint_targets'], previous_action=b['physical']['final_action'],
        feet=b['physical']['foot_contact'], next_feet=a['physical']['foot_contact'],
        present=b['physical']['ball_present'], elapsed=b['physical']['elapsed_s'],
        episode=row['episode_index'], terminated=row['terminated'], timeout=row['timeout'],
        torque=a['auxiliary']['reward_inputs']['applied_torque_all'],
        complete_reward=row['reward_components']['complete5_terminal'] > 0)


def current_start(row):
    """Keep only current state and executor memory; never future labels."""
    return tensor_tree_copy({key: row[key] for key in ('before', 'executor_before', 'observation')}, cpu=True)


class PhysicalLoopReplay:
    """CPU replay with chronological ring storage and reset-safe windows."""
    def __init__(self, package, schema, *, capacity_ticks=16384):
        self.schema = schema
        self.capacity = int(capacity_ticks)
        if self.capacity < 64:
            raise ValueError('Long replay capacity must be at least 64 ticks')
        source_count = len(package['tensors']['state'])
        if source_count > self.capacity:
            raise ValueError('Initialization archive exceeds replay capacity')
        self.required = ('state', 'next_state', 'values', 'valid', 'next_values', 'next_valid',
            'event_state', 'next_event_state', 'targets', 'previous_action', 'feet',
            'next_feet', 'present', 'elapsed', 'episode', 'terminated', 'timeout',
            'torque', 'complete_reward')
        self.data = {key: torch.empty((self.capacity, *package['tensors'][key].shape[1:]),
                                      dtype=package['tensors'][key].dtype) for key in self.required}
        for key, tensor in self.data.items():
            tensor[:source_count].copy_(package['tensors'][key])
        self.count = source_count
        self.write_pos = source_count % self.capacity
        self.streams = torch.zeros(self.capacity, dtype=torch.long)
        self.streams[:source_count] = 1
        self.current_stream = 1
        self.generator = torch.Generator().manual_seed(20261114)
        self.new_starts = []
        self.stage_starts = {name: [] for name in ('prepare', 'approach', 'contact', 'touchdown', 'recovery', 'terminal')}

    def new_stream(self):
        self.current_stream += 1

    def _stage(self, row):
        physical = row['before']['physical']
        elapsed = physical['elapsed_s']
        feet = physical['foot_contact']
        event = row['before']['event_state']
        if bool((event[..., 0:4] > 0).any()):
            return 'contact'
        if bool(feet.any()):
            return 'touchdown' if float(elapsed.max()) < 4.0 else 'recovery'
        if bool((elapsed < 1.6).any()):
            return 'prepare'
        if bool((elapsed >= 7.0).any()):
            return 'terminal'
        return 'approach'

    def append(self, row):
        record = row_tensors(row, self.schema)
        index = self.write_pos
        for key, tensor in self.data.items():
            tensor[index].copy_(record[key].detach().cpu())
        self.streams[index] = self.current_stream
        self.write_pos = (index + 1) % self.capacity
        self.count = min(self.capacity, self.count + 1)
        start = current_start(row)
        stage = self._stage(row)
        self.stage_starts[stage].append(start)
        self.stage_starts[stage] = self.stage_starts[stage][-64:]
        self.new_starts.append(start)
        self.new_starts = self.new_starts[-256:]

    def _ordered(self):
        if self.count < self.capacity:
            indices = torch.arange(self.count)
        else:
            indices = torch.cat((torch.arange(self.write_pos, self.capacity),
                                torch.arange(0, self.write_pos)))
        return {key: value[indices] for key, value in self.data.items()}, self.streams[indices]

    def window(self, steps=16, batch=128, iteration=0, device='cuda:0'):
        d, streams = self._ordered()
        if self.count < steps:
            raise RuntimeError(f'Replay has {self.count} ticks; needs {steps}')
        valid = valid_windows(d['episode'], d['terminated'], d['timeout'], steps)
        stream_window = streams.unfold(0, steps, 1)
        valid &= (stream_window == stream_window[:, :1]).all(-1)[:, None]
        pools = {'uniform': valid.nonzero()}
        for i in range(8):
            event = ((d['next_event_state'][..., i] > d['event_state'][..., i])
                     .unfold(0, steps, 1).any(-1) & valid)
            if event.any():
                pools[str(i)] = event.nonzero()
        chosen, counts = sample_event_windows(pools, batch, self.generator, iteration)
        t, lanes = chosen.unbind(1)
        at = t[None] + torch.arange(steps)[:, None]
        window = {key: value[at, lanes[None]].to(device) for key, value in d.items()}
        return window, lanes.to(device), counts
