"""Compact one archived controller clock at a time, retaining its lane-local state."""
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
