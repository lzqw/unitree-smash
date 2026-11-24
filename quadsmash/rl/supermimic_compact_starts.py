"""Compact and batch native starts while preserving controller clocks."""
import torch


def compact_start(row, selected, launch):
    n=len(row['observation'])
    if selected.shape!=(n,) or launch.shape!=(n,6):
        raise ValueError('Invalid lane alignment in compact imagination start')
    ids=selected.nonzero().flatten()
    if not len(ids):
        raise ValueError('Empty compact imagination start')
    if n!=128:
        if n==len(ids):
            return row,launch
        raise ValueError('Unsupported source controller batch shape')

    def gather(x):
        if isinstance(x,torch.Tensor) and x.ndim and x.shape[0]==n:
            return x.index_select(0,ids.to(x.device))
        if isinstance(x,dict):
            return {k:gather(v) for k,v in x.items()}
        return x

    executor=row['executor_before']
    # Controller cursors index the original plan history. Its per-plan rows are
    # indexed by recorded env_ids and do not share the current lane dimension.
    tensors={k:(v if '/logs/' in k else gather(v)) for k,v in executor['tensors'].items()}
    values={k:(len(ids) if k.endswith('/count') and '/checkpoint/' not in k and v==n else v)
            for k,v in executor['values'].items()}
    snapshot=dict(executor,tensors=tensors,values=values)
    return dict(row,before=gather(row['before']),observation=gather(row['observation']),
                executor_before=snapshot,_actor_lane_mask=torch.ones(len(ids),dtype=torch.bool)),gather(launch)


def merge_physical_starts(states, *, pack_execution=False):
    from .supermimic_g07_memory_preview_v1 import G07BatchMemory, pack_planned_execution
    from .supermimic_physical_imagination_state_v1 import PhysicalImaginationStateV1

    if len(states) == 1:
        return states[0]
    counts = tuple(len(state.physical) for state in states)
    device = states[0].physical.device

    def merge(items, path=()):
        first = items[0]
        name = path[-1] if path else ''
        if isinstance(first, dict):
            if any(item.keys() != first.keys() for item in items[1:]):
                raise ValueError(f'Incompatible start fields at {path}')
            return {key: merge([item[key] for item in items], path + (key,)) for key in first}
        if name not in ('limits', 'command_limits') and isinstance(first, torch.Tensor):
            if all(item.ndim and len(item) == n for item, n in zip(items, counts, strict=True)):
                return torch.cat(items)
        if 'timing' in path and name != '_jump_metric_stride':
            clocks = [value for value, n in zip(items, counts, strict=True) for _ in range(n)]
            return torch.as_tensor(clocks, device=device, dtype=torch.long)
        for item in items[1:]:
            same = torch.equal(first, item) if isinstance(first, torch.Tensor) else first == item
            if not same:
                raise ValueError(f'Incompatible shared physical contract at {path}')
        return first

    branches, branch_counts = [], []
    for state, count in zip(states, counts, strict=True):
        memory = state.executor_memory
        if isinstance(memory, G07BatchMemory):
            branches.extend(memory.branches)
            branch_counts.extend(memory.lane_counts)
        else:
            branches.append(memory)
            branch_counts.append(count)
    executor = (pack_planned_execution(tuple(branches), tuple(branch_counts)) if pack_execution
                else G07BatchMemory(tuple(branches), tuple(branch_counts)))
    return PhysicalImaginationStateV1(
        physical=torch.cat([state.physical for state in states]),
        executor_memory=executor,
        task_memory=merge([state.task_memory for state in states]),
        observation_history=torch.cat([state.observation_history for state in states]),
        observation=torch.cat([state.observation for state in states]),
        clock_goal_context=merge([state.clock_goal_context for state in states]))


def batch_approach_starts(states, *, max_steps, dt, followthrough_ticks, plane_offset_m, pack_execution=False):
    from .supermimic_toi_racket_objective import approach_horizon, flight_intercept

    groups = {}
    for state in states:
        clock = state.clock_goal_context
        forecast = flight_intercept(state, max_steps * dt, plane_offset_m=plane_offset_m)
        horizon = approach_horizon(dict(intercept=forecast), clock['actor_lane_mask'], clock['elapsed'],
                                   maximum=max_steps, dt=dt, followthrough_ticks=followthrough_ticks)
        # Historical relaxed archives add recovery labels that the new phase
        # does not collect. Keep each archive's real auxiliary state.
        recovery = tuple(sorted(state.task_memory['recovery']))
        capture = tuple(state.task_memory['auxiliary']['not_captured'])
        planned = pack_execution and bool(state.executor_memory.reference_controller.planned.all())
        groups.setdefault((horizon, recovery, capture, planned), []).append(state)
    return tuple(merge_physical_starts(groups[key], pack_execution=pack_execution) for key in sorted(groups))
