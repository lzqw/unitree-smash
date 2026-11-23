"""Bounded full-executor snapshots; future outcome labels are sampling-only."""
import torch

from .supermimic_terminal_scoring import APPROACH_BALL_SCORING, SCORING_SCHEMAS

CATEGORIES=('gate_onset','pre_contact','failure_pre','success_near')
SCHEDULE=('gate_onset',)*5+('pre_contact',)*7+('failure_pre',)*5+('success_near',)*3
# Same 20-draw quotas, spread across the cycle so a rejected proposal
# does not immediately replay the very same sparse family with unchanged Adam.
INTERLEAVED_SCHEDULE = (
    'pre_contact', 'gate_onset', 'failure_pre', 'pre_contact', 'success_near',
    'pre_contact', 'gate_onset', 'failure_pre', 'pre_contact', 'gate_onset',
    'success_near', 'pre_contact', 'failure_pre', 'gate_onset', 'pre_contact',
    'failure_pre', 'success_near', 'pre_contact', 'gate_onset', 'failure_pre',
)
BALL_START_QUOTAS = dict(contact_failure=4, low_speed=6, flight_failure=3, success=3)
BALL_START_TIME_QUOTAS = dict(early=5, middle=8, late=3)


def _scale_start_quotas(quotas, count):
    """Allocate the requested batch proportionally, rounding by largest remainder."""
    base = sum(quotas.values())
    allocations = {name: divmod(count * weight, base) for name, weight in quotas.items()}
    rounded = {name: whole for name, (whole, _) in allocations.items()}
    remaining = count - sum(rounded.values())
    for name in sorted(quotas, key=lambda name: -allocations[name][1])[:remaining]:
        rounded[name] += 1
    return rounded


def ball_start_outcome(record):
    if record['approach_ball_terminal']['numerical_abort']:
        return None
    speed = record['outgoing_speed_mps']
    if not record['qualified_airborne_string_contact'] or speed is None:
        return 'contact_failure'
    if 'speed_threshold' in record['rejection_reasons']:
        return 'low_speed'
    return 'success' if record['ball_side_complete_except_recovery'] else 'flight_failure'

def verified_labeled_start_history(older, resumed):
    """Expose labeled snapshots from an older checkpoint on the same replay lineage."""
    for field in ('task_contract', 'policy_execution', 'phase_frozen_actor_sha256', 'phase_anchor_actor_sha256'):
        if older['metadata'][field] != resumed['metadata'][field]:
            raise ValueError(f'Labeled starts use a different {field}')
    if older['config']['wm_holdout_lanes'] != resumed['config']['wm_holdout_lanes']:
        raise ValueError('Labeled starts use a different native holdout')
    if older['global_counters']['accepted'] >= resumed['global_counters']['accepted']:
        raise ValueError('Labeled starts require an earlier checkpoint')
    count = len(older['replay_archives'])
    if not count or older['replay_archives'] != resumed['replay_archives'][:count]:
        raise ValueError('Labeled starts require the same native replay lineage')
    old_receipts = older['replay_archive_receipts']
    current_receipts = resumed['replay_archive_receipts'][:count]
    for old_receipt, current_receipt in zip(old_receipts, current_receipts, strict=True):
        chunks = old_receipt['chunks']
        if chunks != current_receipt['chunks'][:len(chunks)]:
            raise ValueError('Labeled start archive is not a prefix of resumed data')
    bank = older['actor_start_bank']
    if not bank['ball_phase']:
        raise ValueError('Labeled starts must come from the ball-only phase')
    return tuple(entry for entry in bank['entries'] if entry.get('ball_outcomes'))


class ActorStartBank:
    schema='causal-full-executor-start-bank-v1'
    def __init__(self,capacity=32,seed=20261117,*,ball_phase=False,label_reserve=0,excluded_lanes=(),
                 persistent_events=False):
        if not 0 <= label_reserve < capacity or (label_reserve and not ball_phase):
            raise ValueError('Labeled snapshot reserve must fit within the ball-phase bank')
        self.capacity=capacity;self.entries=[];self.seen=0;self.last_gate={};self.last_tick={}
        self.ball_phase=bool(ball_phase)
        self.label_reserve=label_reserve
        self.excluded_lanes=set(excluded_lanes)
        self.generator=torch.Generator().manual_seed(seed)
        self.sampling_history=[]
        self.persistent_events = persistent_events
        self.pending = {}
        self.resolved = {}

    def discard_pending_streams(self, stream):
        # A new native scene cannot finish episodes from an earlier stream.
        self.pending = {key: value for key, value in self.pending.items() if key[0] == stream}

    def _register_pending(self, entry):
        if not self.persistent_events:
            return
        episodes = entry['start']['before']['episode_index']
        for lane in entry['masks']['pre_contact'].nonzero().flatten().tolist():
            if lane in self.excluded_lanes:
                continue
            key = (entry['stream'], lane, int(episodes[lane]))
            frames = self.pending.setdefault(key, [])
            # Keep a short, time-spaced precontact record per actual episode.
            # Entries are referenced, so multiple lanes share their full snapshot.
            if not frames or entry['native_tick'] - frames[-1]['native_tick'] >= 8:
                frames.append(entry)
                del frames[1:-5]

    def _prune_ball_entries(self):
        if len(self.entries) <= self.capacity:
            return
        if not self.label_reserve:
            del self.entries[:-self.capacity]
            return
        recent_count = self.capacity - self.label_reserve
        older = self.entries[:-recent_count]
        recent = self.entries[-recent_count:]

        def labeled_episodes(entry):
            episodes = entry['start']['before']['episode_index']
            return {(entry['stream'], lane, int(episodes[lane]))
                    for lane in entry.get('ball_outcomes', {}) if lane not in self.excluded_lanes}

        covered = set().union(*(labeled_episodes(entry) for entry in recent))
        retained = set()
        for i in range(len(older) - 1, -1, -1):
            labels = labeled_episodes(older[i])
            if labels - covered:
                retained.add(i)
                covered.update(labels)
                if len(retained) == self.label_reserve:
                    break
        for i in range(len(older) - 1, -1, -1):
            if len(retained) == self.label_reserve:
                break
            retained.add(i)
        self.entries[:] = [older[i] for i in sorted(retained)] + recent

    def append(self,start,metadata,*,stream,policy_version,native_tick):
        if 'approach_gate' not in metadata:return
        gate=metadata['approach_gate'].detach().cpu().reshape(-1)
        pre=metadata['had_contact'].detach().cpu().reshape(-1)<.5
        prior=self.last_gate.get(stream,torch.zeros_like(gate));self.last_gate[stream]=gate.clone()
        onset=pre&(((gate>0)&(gate<=.25))|((gate>.25)&(prior<=.25)))
        window = metadata.get('approach_start_eligible', gate > .25)
        active=pre&window.detach().cpu().reshape(-1).bool()
        # Native collection runs under inference mode.  Tensors created by
        # boolean operations there are inference tensors and cannot later be
        # updated when terminal labels arrive.  Materialize ordinary CPU
        # tensors for the mutable mask state kept by this bank.
        def mutable_mask(value):
            # ``empty`` created while an outer inference_mode is active is
            # itself an inference tensor.  Explicitly leave that mode for
            # this small metadata allocation so delayed labels can mutate it.
            with torch.inference_mode(False):
                out=torch.empty(value.shape,dtype=torch.bool)
                out.copy_(value.to(dtype=torch.bool))
            return out
        onset=mutable_mask(onset)
        active=mutable_mask(active)
        if not (onset|active).any():return
        if (self.ball_phase or not onset.any()) and native_tick-self.last_tick.get(stream,-100)<4:return
        self.last_tick[stream]=native_tick;self.seen+=1
        entry=dict(start=start,stream=int(stream),policy_version=int(policy_version),native_tick=int(native_tick),
                   masks=dict(gate_onset=onset,pre_contact=active,
                              failure_pre=mutable_mask(torch.zeros_like(active)),
                              success_near=mutable_mask(torch.zeros_like(active))))
        # Reservoir avoids "last 8 lanes" and retains pre-event snapshots until
        # terminal labels arrive. No snapshot is synthesized by mixing G07 trees.
        if self.ball_phase:
            # Approach outcome labels arrive after the flight. Preserve a
            # recent, time-spaced set of full starts for the current phase.
            self.entries.append(entry)
            self._register_pending(entry)
            self._prune_ball_entries()
        elif len(self.entries)<self.capacity:self.entries.append(entry)
        else:
            slot=int(torch.randint(self.seen,(1,),generator=self.generator))
            if slot<self.capacity:self.entries[slot]=entry
        # Gate edge bookkeeping is needed only for the current stream.
        self.last_gate={stream:self.last_gate[stream]};self.last_tick={stream:self.last_tick[stream]}
    def label_terminal(self,record,stream):
        if record.get('scoring_schema') not in SCORING_SCHEMAS:raise ValueError('Start outcome requires independent native terminal scoring')
        if record.get('ball_side_complete_except_recovery') is None:raise ValueError('Missing ball-side terminal label')
        lane=int(record['env_id']);episode=int(record['episode_index'])
        key = (int(stream), lane, episode)
        frames = self.pending.pop(key, [])
        matched_entries = {}
        for entry in {id(e): e for e in (*self.entries, *frames)}.values():
            if entry['stream']!=stream:continue
            before=entry['start']['before']
            if int(before['episode_index'][lane])!=episode:continue
            eligible=entry['masks']['pre_contact'][lane]
            name='success_near' if record['ball_side_complete_except_recovery'] else 'failure_pre'
            target=entry['masks'][name]
            # Some snapshots are created inside the native collector's
            # inference_mode.  Normalize the destination before the delayed
            # terminal-label write, otherwise PyTorch rejects the in-place
            # update even though the mask is logically mutable metadata.
            if target.is_inference():
                with torch.inference_mode(False):
                    mutable=torch.empty(target.shape,dtype=torch.bool)
                    mutable.copy_(target)
                entry['masks'][name]=target=mutable
            target[lane]=bool(eligible)
            if eligible and record['scoring_schema'] == APPROACH_BALL_SCORING:
                label = ball_start_outcome(record)
                contact = record['contact_time_s']
                elapsed = float(before['physical']['elapsed_s'][lane])
                lead = None if contact is None else round((contact - elapsed) / .01)
                if label is not None and (lead is None or 8 <= lead <= 40):
                    entry.setdefault('ball_outcomes', {})[lane] = (label, lead)
                    matched_entries[entry['native_tick']] = entry
        if self.persistent_events and matched_entries and lane not in self.excluded_lanes:
            terminal = {name: record.get(name) for name in (
                'case_id', 'contact_time_s', 'outgoing_speed_mps', 'native_flight',
                'native_contact', 'ball_side_complete_except_recovery',
                'qualified_airborne_string_contact', 'rejection_reasons', 'behavior_at_terminal')}
            terminal['failure_category'] = ball_start_outcome(record)
            representatives = {}
            for tick in sorted(matched_entries):
                entry = matched_entries[tick]
                lead = entry['ball_outcomes'][lane][1]
                interval = None if lead is None else lead // 8
                representatives.setdefault(interval, entry)
            self.resolved[key] = (tuple(representatives.values()), terminal)
            # Bound each category independently so abundant misses do not
            # evict low-speed or successful reference episodes.
            same_outcome = [k for k, (_, signal) in self.resolved.items()
                            if signal['failure_category'] == terminal['failure_category']]
            if len(same_outcome) > 48:
                del self.resolved[same_outcome[0]]

    def _resolved_starts(self):
        unique = {}
        for (_, lane, _), (frames, _) in self.resolved.items():
            for entry in frames:
                unique.setdefault(id(entry), (entry, set()))[1].add(lane)
        return unique.values()

    def sample_ball_phase(self, eligible_lanes, attempt, *, fallback=(), count=16,
                          shuffle_unlabeled=False, labeled_history=(), excluded_episodes=(),
                          require_labels=False, record_draw=True, reserve_for_proposal=0):
        """Draw distinct episode starts with proportionate outcome and time quotas."""
        if not 1 <= count <= 128:
            raise ValueError('Ball-phase start count must be in [1,128]')
        outcome_quotas = _scale_start_quotas(BALL_START_QUOTAS, count)
        order = torch.randperm(len(self.entries), generator=self.generator).tolist()
        groups = {name: [] for name in BALL_START_QUOTAS}
        unlabeled = []
        entries = [(self.entries[i], False, None) for i in order]
        entries.extend((entry, True, None) for entry in labeled_history)
        entries.extend((entry, True, lanes) for entry, lanes in self._resolved_starts())
        for entry, history, resolved_lanes in entries:
            start = entry['start']
            valid = eligible_lanes(start).cpu() & entry['masks']['pre_contact']
            for lane in valid.nonzero().flatten().tolist():
                if resolved_lanes is not None and lane not in resolved_lanes:
                    continue
                outcome, lead = entry.get('ball_outcomes', {}).get(lane, ('unlabeled', None))
                if history and outcome not in groups:
                    continue
                group = groups.get(outcome, unlabeled)
                group.append((start, lane, outcome, lead))
        for start in fallback:
            for lane in eligible_lanes(start).cpu().nonzero().flatten().tolist():
                unlabeled.append((start, lane, 'unlabeled', None))

        # These lead-time preferences share the same batch as the outcome quotas.
        time_quotas = _scale_start_quotas(BALL_START_TIME_QUOTAS, count)
        timing = tuple(name for name, quota in time_quotas.items() for _ in range(quota))
        timing = timing[attempt % len(timing):] + timing[:attempt % len(timing)]
        seen, selected = set(), {}
        excluded_episodes = set(excluded_episodes)
        actual = {name: 0 for name in (*BALL_START_QUOTAS, 'unlabeled')}
        times = dict(early=0, middle=0, late=0, unknown=0)

        def time_group(lead):
            return 'unknown' if lead is None else 'early' if lead >= 24 else 'middle' if lead >= 12 else 'late'

        def episode_identity(start, lane):
            return (int(start.get('stream_id', start.get('_native_stream', 0))),
                    int(start.get('_lane_id', lane)), int(start['before']['episode_index'][lane]))

        outcome_caps = {}
        if reserve_for_proposal:
            proposal_quotas = _scale_start_quotas(BALL_START_QUOTAS, reserve_for_proposal)
            for name, rows in groups.items():
                support = len({episode_identity(start, lane) for start, lane, _, _ in rows} - excluded_episodes)
                if support < outcome_quotas[name] + proposal_quotas[name]:
                    share = round(support * count / (count + reserve_for_proposal))
                    outcome_caps[name] = min(support - 1, max(1, share)) if support > 1 else 0

        if shuffle_unlabeled:
            by_episode = {}
            for row in unlabeled:
                by_episode.setdefault(episode_identity(row[0], row[1]), []).append(row)
            episode_rows = list(by_episode.values())
            order = torch.randperm(len(episode_rows), generator=self.generator).tolist()
            unlabeled = [episode_rows[i][int(torch.randint(len(episode_rows[i]), (), generator=self.generator))]
                         for i in order]

        def take(candidates, requested):
            for _ in range(requested):
                desired = timing[len(seen)]
                remaining = [row for row in candidates
                             if episode_identity(row[0], row[1]) not in seen
                             and episode_identity(row[0], row[1]) not in excluded_episodes
                             and actual[row[2]] < outcome_caps.get(row[2], count)]
                if not remaining:
                    return
                start, lane, outcome, lead = next(
                    (row for row in remaining if time_group(row[3]) == desired), remaining[0])
                before = start['before']
                key = episode_identity(start, lane)
                seen.add(key)
                if id(start) not in selected:
                    mask = torch.zeros(len(before['episode_index']), dtype=torch.bool)
                    selected[id(start)] = (dict(start, _actor_lane_mask=mask), [])
                row, details = selected[id(start)]
                row['_actor_lane_mask'][lane] = True
                details.append([key[1], key[2], int(before['elapsed_tick'][lane]), outcome, lead, key[0]])
                actual[outcome] += 1
                times[time_group(lead)] += 1

        for name, quota in outcome_quotas.items():
            take(groups[name], quota)
        take([row for group in groups.values() for row in group] + ([] if require_labels else unlabeled),
             len(timing) - len(seen))
        report = dict(requested_by_outcome=outcome_quotas, actual_by_outcome=actual,
                      outcome_shortfall={name:max(0, quota-actual[name]) for name,quota in outcome_quotas.items()},
                      lead_time_counts=times, effective_lanes=len(seen), missing_episodes=len(timing)-len(seen),
                      records=[dict(lane_episode_tick=details) for _, details in selected.values()])
        if shuffle_unlabeled:
            report['unlabeled_pool_episodes'] = len(unlabeled)
        if reserve_for_proposal:
            report['scarce_outcome_caps'] = outcome_caps
        if self.ball_phase and record_draw:
            self.sampling_history.append(report)
        return [row for row, _ in selected.values()], report

    def sample_proposal_and_guard(self, eligible_lanes, attempt, proposal_count, guard_count, *, labeled_history=()):
        cache = {}
        def eligible(start):
            if id(start) not in cache:
                cache[id(start)] = eligible_lanes(start)
            return cache[id(start)]

        options = dict(require_labels=True, labeled_history=labeled_history, record_draw=False)
        settings = dict.fromkeys(((proposal_count, guard_count), (32, 16)))
        for n, g in settings:
            guard, guard_report = self.sample_ball_phase(
                eligible, attempt, count=g, reserve_for_proposal=n, **options)
            if guard_report['effective_lanes'] != g:
                continue
            forbidden = {(stream, lane, episode) for record in guard_report['records']
                         for lane, episode, _, _, _, stream in record['lane_episode_tick']}
            proposal, report = self.sample_ball_phase(
                eligible, attempt + g, count=n, excluded_episodes=forbidden, **options)
            if report['effective_lanes'] != n:
                continue
            report.update(guard=guard_report, disjoint_native_episodes=True,
                          requested_proposal_lanes=proposal_count, requested_guard_lanes=guard_count)
            self.sampling_history.append(report)
            return proposal, guard, report
        raise RuntimeError('Need at least 48 eligible labeled native episodes for disjoint Approach v2 batches')

    def sample(self, attempt, *, eligible_lanes=None, interleaved=False):
        schedule = INTERLEAVED_SCHEDULE if interleaved else SCHEDULE
        requested = schedule[(attempt - 1) % len(schedule)]
        checks={}
        categories=dict.fromkeys((requested,'pre_contact','gate_onset','failure_pre','success_near'))
        for actual in categories:
            entries=[entry for entry in self.entries if entry['masks'][actual].any()]
            if not entries:
                continue
            if eligible_lanes is None:
                order=[int(torch.randint(len(entries),(1,),generator=self.generator))]
            else:
                order=torch.randperm(len(entries),generator=self.generator).tolist()
            for index in order:
                entry=entries[index]
                mask=entry['masks'][actual]
                if eligible_lanes is not None:
                    key=id(entry)
                    if key not in checks:
                        checks[key]=eligible_lanes(entry['start']).detach().reshape(-1).cpu()
                    valid=checks[key]
                    if valid.dtype!=torch.bool or valid.shape!=mask.shape:
                        raise ValueError('Start eligibility must identify each original lane')
                    mask=mask & valid
                    if not mask.any():
                        continue
                row=dict(entry['start'],_actor_lane_mask=mask.clone())
                report=dict(requested=requested,actual=actual,shortfall=int(actual!=requested),
                            effective_lanes=int(mask.sum()),policy_version=entry['policy_version'],
                            native_tick=entry['native_tick'],stream_id=entry['stream'],outcomes_are_sampling_only=True)
                if eligible_lanes is not None:
                    report['records_checked_for_objective']=len(checks)
                self.sampling_history.append(report)
                return row,report
        report=dict(requested=requested,actual=None,shortfall=1)
        if eligible_lanes is not None:
            report['records_checked_for_objective']=len(checks)
        self.sampling_history.append(report)
        return None,report
    def candidate_lanes_by_start(self):
        """Union of selectable categories for each current-state record."""
        candidates = {}
        for entry in self.entries:
            key = id(entry['start'])
            mask = torch.stack([entry['masks'][name] for name in CATEGORIES]).any(0)
            if key in candidates:
                candidates[key] |= mask
            else:
                candidates[key] = mask
        return candidates

    def same_category_starts(self, category, attempt):
        entries=self.entries
        if category not in CATEGORIES or not entries:
            return
        offset=(attempt-1)%len(entries)
        for entry in entries[offset:]+entries[:offset]:
            mask=entry['masks'][category]
            if mask.any():
                yield entry['start'],mask

    def sampling_report(self, since=0):
        rows=self.sampling_history[since:]
        if self.ball_phase:
            return dict(samples=len(rows),
                requested_by_outcome={name:sum(row['requested_by_outcome'][name] for row in rows) for name in BALL_START_QUOTAS},
                actual_by_outcome={name:sum(row['actual_by_outcome'][name] for row in rows) for name in (*BALL_START_QUOTAS,'unlabeled')},
                shortfall_by_outcome={name:sum(row['outcome_shortfall'][name] for row in rows) for name in BALL_START_QUOTAS},
                lead_time_counts={name:sum(row['lead_time_counts'][name] for row in rows) for name in ('early','middle','late','unknown')})
        return dict(samples=len(rows),actual_counts={c:sum(r['actual']==c for r in rows) for c in CATEGORIES},
            requested_counts={c:sum(r['requested']==c for r in rows) for c in CATEGORIES},
            shortfall=sum(r['shortfall'] for r in rows),
            unlabeled_lane_snapshots=sum(int((e['masks']['pre_contact']&~e['masks']['failure_pre']&~e['masks']['success_near']).sum()) for e in self.entries))
    def state_dict(self):
        frames = {}
        def index(items):
            keys = [(entry['stream'], entry['native_tick']) for entry in items]
            frames.update(zip(keys, items, strict=True))
            return keys
        pending = {k: index(rows) for k, rows in self.pending.items()}
        resolved = {k: (index(rows), label) for k, (rows, label) in self.resolved.items()}
        return dict(schema=self.schema,capacity=self.capacity,entries=self.entries,seen=self.seen,
                    ball_phase=self.ball_phase,sampling_history=self.sampling_history,last_gate=self.last_gate,last_tick=self.last_tick,
                    event_frames=frames, pending_frames=pending, resolved_events=resolved,
                    rng=self.generator.get_state())
    def load_state_dict(self,state):
        if state['schema']!=self.schema or state['capacity']!=self.capacity:raise ValueError('Actor start bank schema/capacity mismatch')
        if bool(state.get('ball_phase',False))!=self.ball_phase:raise ValueError('Actor start bank phase differs; a new phase requires its own start library')
        if len(state['entries'])>self.capacity:raise ValueError('Actor start bank exceeds capacity')
        self.entries=state['entries'];self.seen=int(state['seen']);self.last_gate=state['last_gate'];self.last_tick=state['last_tick']
        self.sampling_history=list(state.get('sampling_history',[]))
        if state.get('resolved_events') and not self.persistent_events:
            raise ValueError('Restoring an event bank requires persistent event collection')
        frames = state.get('event_frames', {}) | {
            (entry['stream'], entry['native_tick']): entry for entry in self.entries}
        self.pending = {k: [frames[i] for i in ids] for k, ids in state.get('pending_frames', {}).items()}
        self.resolved = {k: (tuple(frames[i] for i in ids), label)
                         for k, (ids, label) in state.get('resolved_events', {}).items()}
        self.generator.set_state(state['rng'].cpu())


def pool_toi_starts(first, alternatives, eligible_lanes, target, attempt):
    """Retain independent lane episodes and each snapshot's native G07 execution state."""
    seen=set()
    starts=[]
    layout=[]

    def episode_key(row, lane):
        return int(row.get('_lane_id', lane)), int(row['before']['episode_index'][lane])

    def take(row, mask):
        mask=mask.detach().cpu().bool().reshape(-1)
        keep=torch.zeros_like(mask)
        before=row['before']
        lanes=mask.nonzero().flatten().tolist()
        offset=(attempt-1)%max(1,len(lanes))
        records=[]
        for lane in lanes[offset:]+lanes[:offset]:
            key=episode_key(row, lane)
            if key in seen:
                continue
            seen.add(key)
            keep[lane]=True
            records.append([*key,int(before['elapsed_tick'][lane])])
            if len(seen)==target:
                break
        if records:
            starts.append(dict(row,_actor_lane_mask=keep))
            layout.append(dict(lane_episode_tick=records))

    if target<1 or attempt<1:
        raise ValueError('Positive TOI batch size and proposal index required')
    take(first,first['_actor_lane_mask'])
    for row,category_mask in alternatives:
        if len(seen)==target:
            break
        if row['before'] is first['before']:
            continue
        # Check only membership here. Filtering individual lanes before take()
        # would change its rotation and bias the selected episodes.
        if category_mask is not None:
            lanes = category_mask.reshape(-1).nonzero().flatten().tolist()
            if all(episode_key(row, lane) in seen for lane in lanes):
                continue
        valid=eligible_lanes(row).detach().cpu().reshape(-1)
        if category_mask is not None:
            valid=valid & category_mask
        take(row,valid)
    if not starts:
        raise ValueError('TOI batch has no eligible episode')
    return starts,layout
