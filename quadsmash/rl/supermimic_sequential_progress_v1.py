"""Explicit per-block progress; optimizer accepts never imply native acceptance."""
import copy

class SequentialProgress:
    schema='sequential-proposal-progress-v1'
    def __init__(self):self.cycles={}
    @classmethod
    def for_run(cls, saved, *, fresh):
        progress = cls()
        if saved is not None and not fresh:
            progress.load_state_dict(saved)
        return progress

    def block(self,cycle,head,target,limit):
        key=f'{cycle}/{head}'
        if key not in self.cycles:self.cycles[key]=dict(target=target,limit=limit,attempts=0,accepted=0,rejected=0,backtracked=0,status='new',refresh_at=0,pending_refresh=False)
        row=self.cycles[key]
        if (row['target'],row['limit'])!=(target,limit):raise ValueError('Resume changes block budget')
        return row
    @staticmethod
    def finish_approach_refresh(row, updates):
        if not row['pending_refresh'] or row['accepted']<=row['refresh_at']:
            raise ValueError('No new Approach refresh to commit')
        row.setdefault('refresh_counts_complete', row['refresh_at']==0)
        for key, count in zip(('refresh_world_updates','refresh_critic_updates'), updates, strict=True):
            row[key]=row.get(key,0)+count
        row.update(refresh_at=row['accepted'],pending_refresh=False)

    def cycle_refresh_counts(self, cycle, recovery_world, recovery_critic):
        row=self.cycles.get(f'{cycle}/approach',{})
        known=row.get('refresh_counts_complete',not row.get('refresh_at'))
        return dict(refresh_counts_complete=known,
                    refresh_world_updates=recovery_world+row.get('refresh_world_updates',0) if known else None,
                    refresh_critic_updates=recovery_critic+row.get('refresh_critic_updates',0) if known else None)

    def state_dict(self):return dict(schema=self.schema,cycles=copy.deepcopy(self.cycles))
    def load_state_dict(self,state):
        if state['schema']!=self.schema:raise ValueError('Sequential progress schema mismatch')
        self.cycles=copy.deepcopy(state['cycles'])
        for row in self.cycles.values():
            if not 0<=row['accepted']<=row['attempts']<=row['limit'] or row['accepted']>row['target']:raise ValueError('Corrupt block progress')
    def complete(self):
        cycles={k.split('/')[0] for k in self.cycles}
        return bool(cycles) and all(all(f'{c}/{h}' in self.cycles and self.cycles[f'{c}/{h}']['status']=='done' for h in ('recovery','approach')) for c in cycles)
