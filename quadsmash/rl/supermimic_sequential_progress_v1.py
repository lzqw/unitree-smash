"""Explicit per-block progress; optimizer accepts never imply native acceptance."""
import copy

class SequentialProgress:
    schema='sequential-proposal-progress-v1'
    def __init__(self):self.cycles={}
    def block(self,cycle,head,target,limit):
        key=f'{cycle}/{head}'
        if key not in self.cycles:self.cycles[key]=dict(target=target,limit=limit,attempts=0,accepted=0,rejected=0,backtracked=0,status='new',refresh_at=0,pending_refresh=False)
        row=self.cycles[key]
        if (row['target'],row['limit'])!=(target,limit):raise ValueError('Resume changes block budget')
        return row
    def state_dict(self):return dict(schema=self.schema,cycles=copy.deepcopy(self.cycles))
    def load_state_dict(self,state):
        if state['schema']!=self.schema:raise ValueError('Sequential progress schema mismatch')
        self.cycles=copy.deepcopy(state['cycles'])
        for row in self.cycles.values():
            if not 0<=row['accepted']<=row['attempts']<=row['limit'] or row['accepted']>row['target']:raise ValueError('Corrupt block progress')
    def complete(self):
        cycles={k.split('/')[0] for k in self.cycles}
        return bool(cycles) and all(all(f'{c}/{h}' in self.cycles and self.cycles[f'{c}/{h}']['status']=='done' for h in ('recovery','approach')) for c in cycles)
