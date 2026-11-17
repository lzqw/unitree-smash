"""Reset-safe, long-running physical replay with a chronological ring buffer."""
import torch
from .supermimic_physical_dataset_v1 import encode_physical, valid_windows
from .supermimic_physical_world_v2 import sample_event_windows
from .supermimic_physical_capture_v1 import tensor_tree_copy
from .supermimic_approach_data_v1 import FIELDS, approach_window_masks


SOURCE_NAMES = ('recent_online', 'older_online', 'initial_archive')
SOURCE_CODES = {'initial_archive': 0, 'online': 1}
STAGE_NAMES = ('uniform', 'pre_touchdown', 'touchdown', 'post_touchdown',
               'recovery_failure', 'stable_recovery')


def _stage_ids_from_tensors(data):
    """Classify every recorded tick/lane without using a batch-wide ``any``.

    The labels are replay metadata only.  They are never fed to the actor and
    may therefore use the event that occurs inside a sampled window.  A lane
    is labelled from its own contact/event state so an event in another lane
    cannot turn a normal window into a touchdown or failure window.
    """
    event_delta = data['next_event_state'] > data['event_state']
    feet = data['feet'].bool().any(-1)
    next_feet = data['next_feet'].bool().any(-1)
    jump_seen = data['event_state'][..., 0] > 0.5
    touchdown = jump_seen & (~feet) & next_feet
    body_ground = event_delta[..., 5] | event_delta[..., 7]
    stable = event_delta[..., 4] | data['complete_reward'].bool()
    elapsed = data['elapsed']
    # Numeric codes are intentionally private to replay metadata.
    # 0 prepare, 1 approach, 2 pre-touchdown, 3 post-touchdown,
    # 4 touchdown, 5 recovery failure, 6 stable recovery.
    stage = torch.where(elapsed < 1.6, torch.zeros_like(elapsed, dtype=torch.long),
            torch.where(elapsed < 3.5, torch.ones_like(elapsed, dtype=torch.long),
            torch.full_like(elapsed, 2, dtype=torch.long)))
    # Replace the coarse clock label once the causal jump/flight evidence is
    # available.  Initial standing feet therefore remain prepare/approach;
    # post-touchdown requires an observed jump followed by current support.
    stage = torch.where(jump_seen & (~feet), torch.full_like(stage, 2), stage)
    stage = torch.where(jump_seen & feet, torch.full_like(stage, 3), stage)
    stage = torch.where(touchdown, torch.full_like(stage, 4), stage)
    stage = torch.where(stable, torch.full_like(stage, 6), stage)
    stage = torch.where(body_ground, torch.full_like(stage, 5), stage)
    return stage


def _stage_window_masks(data, valid, steps, stage_ids=None):
    """Return event-aligned masks for valid [time,lane] window starts."""
    if stage_ids is None:
        stage_ids = _stage_ids_from_tensors(data)
    event_delta = data['next_event_state'] > data['event_state']
    feet = data['feet'].bool().any(-1)
    next_feet = data['next_feet'].bool().any(-1)
    jump_seen = data['event_state'][..., 0] > 0.5
    touchdown = (jump_seen & (~feet) & next_feet).unfold(0, steps, 1).any(-1)
    failure = (event_delta[..., 5] | event_delta[..., 7]).unfold(0, steps, 1).any(-1)
    stable = (event_delta[..., 4] | data['complete_reward'].bool()).unfold(0, steps, 1).any(-1)
    stage_windows = stage_ids.unfold(0, steps, 1)
    pre = (stage_windows == 2).any(-1)
    post = (stage_windows == 3).any(-1) & ~touchdown
    return {
        'pre_touchdown': pre & valid,
        'touchdown': touchdown & valid,
        'post_touchdown': post & valid,
        'recovery_failure': failure & valid,
        'stable_recovery': stable & valid,
    }


def _sample_rows(pool, count, generator):
    if pool is None:
        return None
    if count <= 0:
        return pool[:0]
    if len(pool) == 0:
        return None
    # Prefer distinct windows; replacement is explicit only when scarce.
    order=torch.randperm(len(pool),generator=generator)
    if count<=len(pool):return pool[order[:count]]
    return torch.cat((pool[order],pool[torch.randint(len(pool),(count-len(pool),),generator=generator)]))


def sample_stratified_windows(stage_pools, source_labels, batch, generator,
                              *, source_quota=None, stage_quota=None):
    """Sample windows with auditable source × stage quotas.

    Prefer distinct windows within each cell; repeat only when it is scarce.
    If a requested source/stage cell is empty, the sampler fills from the
    remaining valid cells and reports the shortfall instead of pretending the
    quota was met.
    """
    if batch < 1:
        raise ValueError('batch must be positive')
    source_quota = dict(source_quota or {})
    stage_quota = dict(stage_quota or {})
    source_quota = {k: int(v) for k, v in source_quota.items() if int(v) > 0}
    stage_quota = {k: int(v) for k, v in stage_quota.items() if int(v) > 0}
    if sum(source_quota.values()) > batch or sum(stage_quota.values()) > batch:
        raise ValueError('Requested quota exceeds batch size')
    # Stage pools contain [time,lane] pairs.  Use source at the window start;
    # valid_windows already guarantees that the stream remains unchanged.
    pair_pools = {}
    for stage, pairs in stage_pools.items():
        if len(pairs) == 0:
            continue
        pair_pools[stage] = {}
        for source in source_quota or ('all',):
            if source == 'all':
                pair_pools[stage][source] = pairs
                continue
            mask = source_labels[pairs[:, 0], pairs[:, 1]] == source
            pair_pools[stage][source] = pairs[mask]

    names_stage = list(stage_quota) or ['uniform']
    names_source = list(source_quota) or ['all']
    if 'uniform' not in pair_pools:
        union = [pairs for pairs in stage_pools.values() if len(pairs)]
        if union:
            all_pairs = torch.cat(union)
            pair_pools['uniform'] = {'all': all_pairs}
            for source in names_source:
                if source != 'all':
                    pair_pools['uniform'][source] = all_pairs[
                        source_labels[all_pairs[:, 0], all_pairs[:, 1]] == source]
    if not stage_quota:
        stage_quota = {'uniform': batch}
    selected = []
    actual_stage = {name: 0 for name in names_stage}
    actual_source = {name: 0 for name in names_source}
    stage_shortfall = {name: 0 for name in names_stage}
    # Allocate the requested quota matrix proportionally, then fill each
    # deficit from any available cell.  This is deterministic apart from the
    # seeded row selection and preserves both quota dimensions when possible.
    remaining_source = {name: source_quota.get(name, batch) for name in names_source}
    for stage, want_stage in stage_quota.items():
        remaining_stage = int(want_stage)
        for source in names_source:
            if remaining_stage <= 0:
                break
            target = min(remaining_stage, remaining_source[source])
            if target <= 0:
                continue
            pool = pair_pools.get(stage, {}).get(source)
            rows = _sample_rows(pool, target, generator)
            if rows is None:
                continue
            selected.append(rows)
            actual_stage[stage] = actual_stage.get(stage, 0) + target
            actual_source[source] = actual_source.get(source, 0) + target
            remaining_stage -= target
            remaining_source[source] -= target
        if remaining_stage:
            stage_shortfall[stage] += remaining_stage
    # Fill the rest using the union of valid cells, reporting that these are
    # fallback samples rather than silently assigning them to a quota.
    fallback = []
    for stage, by_source in pair_pools.items():
        for pool in by_source.values():
            if len(pool):
                fallback.append(pool)
    if not fallback:
        raise ValueError('No valid windows available for stratified sampling')
    union = torch.unique(torch.cat(fallback),dim=0)
    used = sum(len(x) for x in selected)
    if used < batch:
        selected.append(_sample_rows(union, batch - used, generator))
    chosen = torch.cat(selected, dim=0)
    # Keep the returned batch size exact and deterministic after fallback.
    if len(chosen) > batch:
        chosen = chosen[:batch]
    if source_quota:
        chosen_source = source_labels[chosen[:, 0], chosen[:, 1]]
        actual_source = {name: int((chosen_source == name).sum()) for name in names_source}
    # A window may satisfy more than one event category.  Report this
    # post-selection coverage separately from the allocator's requested
    # category counts so TensorBoard cannot be mistaken for an exact quota
    # histogram after fallback sampling.
    actual_event_coverage={}
    for stage, pairs in stage_pools.items():
        stride=source_labels.shape[1]
        actual_event_coverage[stage]=int(torch.isin(chosen[:,0]*stride+chosen[:,1],
                                                   pairs[:,0]*stride+pairs[:,1]).sum())
    return chosen, dict(source_counts=actual_source, stage_counts=actual_stage,
                        source_requested=source_quota, stage_requested=stage_quota,
                        source_shortfall={name: max(0, source_quota.get(name, 0)-actual_source.get(name, 0))
                                          for name in names_source},
                        stage_shortfall=stage_shortfall,
                        actual_event_coverage=actual_event_coverage,
                        fallback_count=max(0, batch-sum(actual_stage.values())))


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
        # Episode identity is part of each before/after state record in the
        # native archive; the row envelope only carries termination flags.
        episode=b['episode_index'], terminated=row['terminated'], timeout=row['timeout'],
        torque=a['auxiliary']['reward_inputs']['applied_torque_all'],
        complete_reward=row['reward_components']['complete5_terminal'] > 0)


def current_start(row):
    """Keep only current state and executor memory; never future labels."""
    return tensor_tree_copy({key: row[key] for key in ('before', 'executor_before', 'observation')}, cpu=True)


def _slice_lane(value, lane, lanes):
    """Slice one lane from a captured start without changing scalar metadata."""
    if torch.is_tensor(value):
        if value.ndim and value.shape[0] == lanes:
            return value[lane:lane + 1].clone()
        return value.clone()
    if isinstance(value, dict):
        return {key: _slice_lane(item, lane, lanes) for key, item in value.items()}
    if isinstance(value, list):
        return [_slice_lane(item, lane, lanes) for item in value]
    if isinstance(value, tuple):
        return tuple(_slice_lane(item, lane, lanes) for item in value)
    return value


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
        self.lanes = int(package['tensors']['state'].shape[1])
        self.stage_ids = torch.empty((self.capacity, self.lanes), dtype=torch.long)
        self.stage_ids[:source_count].copy_(_stage_ids_from_tensors(package['tensors']))
        self.source_ids = torch.zeros((self.capacity, self.lanes), dtype=torch.long)
        self.policy_versions = torch.zeros((self.capacity, self.lanes), dtype=torch.long)
        self.native_ticks = torch.arange(self.capacity, dtype=torch.long)
        self.approach_metadata = {k: torch.full((self.capacity, self.lanes), float('nan'))
            for k in FIELDS + tuple('next_'+k for k in FIELDS[2:])}
        self._native_tick_counter = source_count
        self.count = source_count
        self.write_pos = source_count % self.capacity
        self.streams = torch.zeros(self.capacity, dtype=torch.long)
        self.streams[:source_count] = 1
        self.current_stream = 1
        self.current_source = 'initial_archive'
        self.current_policy_version = 0
        self.generator = torch.Generator().manual_seed(20261114)
        self.new_starts = []
        self.stage_starts = {name: [] for name in
                             ('prepare', 'approach', 'pre_touchdown', 'post_touchdown',
                              'touchdown', 'recovery_failure', 'stable_recovery','approach_active')}

    def new_stream(self, *, source='online', policy_version=None):
        self.current_stream += 1
        if source not in SOURCE_CODES:
            raise ValueError(f'Unknown replay source: {source}')
        self.current_source = source
        if policy_version is not None:
            self.current_policy_version = int(policy_version)

    def _stage(self, row):
        """Return per-lane metadata; no lane may inherit another lane's event."""
        record = row_tensors(row, self.schema)
        ids = _stage_ids_from_tensors(record)
        names = ('prepare', 'approach', 'pre_touchdown', 'post_touchdown',
                 'touchdown', 'recovery_failure', 'stable_recovery')
        ids = ids.reshape(-1)
        return [names[int(value)] for value in ids], ids

    def append(self, row, *, retain_start=True, policy_version=None, native_tick=None, approach_metadata=None):
        """Append one native transition.

        Restoring an old checkpoint must rebuild the ring tensors, but does not
        need to retain every historical full observation/executor graph as a
        proposal start.  Keeping those graphs for every archived row can grow
        memory without bound during resume.  New native rows retain starts by
        default; checkpoint restore can opt out and use the fixed start cache.
        """
        record = row_tensors(row, self.schema)
        index = self.write_pos
        for key, tensor in self.data.items():
            tensor[index].copy_(record[key].detach().cpu())
        stages, stage_ids = self._stage(row)
        self.stage_ids[index].copy_(stage_ids)
        self.source_ids[index].fill_(SOURCE_CODES[self.current_source])
        self.policy_versions[index].fill_(int(self.current_policy_version if policy_version is None else policy_version))
        if native_tick is None:
            native_tick = int(self._native_tick_counter)
        self.native_ticks[index] = int(native_tick)
        self._native_tick_counter = max(self._native_tick_counter + 1, int(native_tick) + 1)
        self.streams[index] = self.current_stream
        for key, values in self.approach_metadata.items():
            supplied=(approach_metadata or row.get('sampling_metadata',{})).get(key)
            values[index].copy_(supplied.detach().cpu() if supplied is not None else torch.full((self.lanes,),float('nan')))
        self.write_pos = (index + 1) % self.capacity
        self.count = min(self.capacity, self.count + 1)
        if not retain_start:
            return
        start = current_start(row)
        # Full executor/observation trees are expensive.  Keep a small recent
        # diagnostic pool; fixed verified pools remain available for each
        # training block, so long runs do not grow memory with every tick.
        # Preserve the full G07 execution graph. A mask selects training lanes;
        # never slice arbitrary tensors/scalars/aliases from a 128-lane template.
        for stage in set(stages):
            mask = torch.tensor([x == stage for x in stages])
            entry = dict(start, _actor_lane_mask=mask, _source_lanes=self.lanes)
            self.stage_starts.setdefault(stage, []).append(entry)
            self.stage_starts[stage] = self.stage_starts[stage][-16:]
        gate_mask=(self.approach_metadata['approach_gate'][index]>.25)&(self.approach_metadata['had_contact'][index]<.5)
        if gate_mask.any():
            self.stage_starts['approach_active'].append(dict(start,_actor_lane_mask=gate_mask.clone(),_source_lanes=self.lanes))
            self.stage_starts['approach_active']=self.stage_starts['approach_active'][-16:]
        self.new_starts.append(start)
        self.new_starts = self.new_starts[-32:]

    def _ordered(self):
        if self.count < self.capacity:
            indices = torch.arange(self.count)
        else:
            indices = torch.cat((torch.arange(self.write_pos, self.capacity),
                                torch.arange(0, self.write_pos)))
        self._ordered_indices=indices
        return ({key: value[indices] for key, value in self.data.items()},
                self.streams[indices], self.stage_ids[indices],
                self.source_ids[indices], self.policy_versions[indices],
                self.native_ticks[indices])

    def window(self, steps=16, batch=128, iteration=0, device='cuda:0',
               *, source_quota=None, stage_quota=None,
               recent_policy_window=32, recent_tick_window=1024, head='recovery'):
        d, streams, stage_ids, source_ids_all, policy_versions, native_ticks = self._ordered()
        if self.count < steps:
            raise RuntimeError(f'Replay has {self.count} ticks; needs {steps}')
        valid = valid_windows(d['episode'], d['terminated'], d['timeout'], steps)
        stream_window = streams.unfold(0, steps, 1)
        valid &= (stream_window == stream_window[:, :1]).all(-1)[:, None]
        # Source is attached to the recorded transition, then refined into
        # recent/older online strata by stream id.  Keep this tensor defined
        # for both legacy and stratified callers; the old path still reports
        # the same metadata without requiring quotas.
        source_ids = source_ids_all[:valid.shape[0]]
        online_mask = source_ids != 0
        online_policy = policy_versions[:valid.shape[0]][online_mask]
        latest_policy = int(online_policy.max()) if online_policy.numel() else 0
        latest_tick = int(native_ticks[:valid.shape[0]][online_mask.any(-1)].max()) if online_mask.any(-1).any() else 0
        recent = online_mask & (policy_versions[:valid.shape[0]] >= max(0, latest_policy-int(recent_policy_window)))
        recent &= native_ticks[:valid.shape[0], None] >= max(0, latest_tick-int(recent_tick_window))
        source_codes = torch.where(source_ids == 0, torch.zeros_like(source_ids),
                                   torch.where(recent, torch.ones_like(source_ids),
                                               torch.full_like(source_ids, 2)))
        pools = {'uniform': valid.nonzero()}
        for i in range(8):
            event = ((d['next_event_state'][..., i] > d['event_state'][..., i])
                     .unfold(0, steps, 1).any(-1) & valid)
            if event.any():
                pools[str(i)] = event.nonzero()
        if source_quota or stage_quota:
            if head == 'approach':
                meta={key:value[self._ordered_indices] for key,value in self.approach_metadata.items()}
                meta['completed_miss']=torch.zeros_like(d['terminated'])
                # Build labels only from a completed episode in this stream.
                for lane in range(self.lanes):
                    ends=torch.where(d['terminated'][:,lane]|d['timeout'][:,lane])[0]
                    for end in ends.tolist():
                        same=(d['episode'][:,lane]==d['episode'][end,lane])&(streams==streams[end])
                        same &= torch.arange(self.count)<=end
                        if torch.isfinite(meta['next_qualified'][end,lane]) and meta['next_qualified'][end,lane]<.5:
                            meta['completed_miss'][:,lane] |= same
                stage_pools=approach_window_masks(meta,valid,steps)
            else:
                stage_pools = _stage_window_masks(d, valid, steps, stage_ids=stage_ids)
            # Preserve a named uniform category for requested batches that do
            # not have enough rare event windows.
            stage_pools['uniform'] = valid
            # Torch has no string tensor; use integer source codes internally.
            pairs = {}
            for name, mask in stage_pools.items():
                pairs[name] = mask.nonzero()
            requested_sources = source_quota or {}
            numeric_source_quota = {
                {'initial_archive': 0, 'recent_online': 1, 'older_online': 2}[name]: int(count)
                for name, count in requested_sources.items()
                if name in ('initial_archive', 'recent_online', 'older_online')
            }
            chosen, counts = sample_stratified_windows(
                pairs, source_codes, batch, self.generator,
                source_quota=numeric_source_quota,
                stage_quota=stage_quota or {})
            source_names = {0: 'initial_archive', 1: 'recent_online', 2: 'older_online', 'all': 'all'}
            for key in ('source_counts', 'source_requested', 'source_shortfall'):
                counts[key] = {source_names.get(name, str(name)): value
                               for name, value in counts[key].items()}
        else:
            chosen, counts = sample_event_windows(pools, batch, self.generator, iteration)
        t, lanes = chosen.unbind(1)
        at = t[None] + torch.arange(steps)[:, None]
        window = {key: value[at, lanes[None]].to(device) for key, value in d.items()}
        if head=='approach' and (source_quota or stage_quota):
            counts['approach_missing_fields']={key:int((~torch.isfinite(value[at,lanes[None]])).sum())
                for key,value in meta.items() if value.is_floating_point()}
        start_episode = d['episode'][t, lanes]
        start_stream = streams[t]
        start_source = source_codes[t, lanes]
        _,episode_counts=torch.unique(torch.stack((start_stream,lanes,start_episode),1),dim=0,return_counts=True)
        counts.update(dict(unique_window_count=int(torch.unique(chosen,dim=0).shape[0]),window_count=int(len(chosen)),
                           unique_episode_count=int(torch.unique(torch.stack((start_stream,lanes,start_episode),1),dim=0).shape[0]),
                           unique_stream_count=int(torch.unique(start_stream).numel()),
                           unique_source_count=int(torch.unique(start_source).numel()),
                           repeated_episode_max=int(episode_counts.max().item())
                           if start_episode.numel() else 0,
                           source_counts={
                               'initial_archive': int((start_source == 0).sum()),
                               'recent_online': int((start_source == 1).sum()),
                               'older_online': int((start_source == 2).sum()),
                           },
                           policy_version_min=int(policy_versions[t, lanes].min().item()) if len(chosen) else 0,
                           policy_version_max=int(policy_versions[t, lanes].max().item()) if len(chosen) else 0,
                           native_tick_min=int(native_ticks[t].min().item()) if len(chosen) else 0,
                           native_tick_max=int(native_ticks[t].max().item()) if len(chosen) else 0,
                           source_metadata='recorded source + policy-version age + native-tick age'))
        return window, lanes.to(device), counts
