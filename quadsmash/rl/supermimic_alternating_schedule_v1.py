"""Bounded short-block orchestration. No simulator, policy or optimizer imports.

One refresh observes the combined actor and feeds BOTH heads. A checkpoint
inside an interrupted refresh is not a transactionally complete training state:
resume the durable pre-refresh checkpoint, never repeat partial optimizer work.
"""
import copy

HEADS = ('recovery', 'approach')


def make_plan(recovery, approach, recovery_proposals, approach_proposals, *,
              block_size=8, cycles=1, refresh_mode='pair'):
    if refresh_mode not in ('pair', 'per-head'):
        raise ValueError('Unknown refresh mode')
    values=(recovery, approach, recovery_proposals, approach_proposals, block_size, cycles)
    if any(type(x) is not int or x < 1 for x in values):
        raise ValueError('Alternating budgets must be positive integers')
    if recovery != approach:
        raise ValueError('Paired schedule requires equal per-cycle head targets')
    if recovery_proposals < recovery or approach_proposals < approach:
        raise ValueError('Proposal limit cannot be below effective-update target')
    sizes=[min(block_size, recovery-i) for i in range(0, recovery, block_size)]
    # Distribute only the declared surplus; no hidden proposal multiplication.
    limits={}
    for head,limit in zip(HEADS,(recovery_proposals,approach_proposals)):
        extra=limit-recovery
        limits[head]=[n+extra//len(sizes)+int(i<extra%len(sizes)) for i,n in enumerate(sizes)]
    events=[]
    for cycle in range(cycles):
        for pair,n in enumerate(sizes):
            key=f'cycle{cycle:03d}_pair{pair:03d}'
            for head in HEADS:
                events.append(dict(id=key+'_'+head,kind='train',cycle=cycle,pair=pair,
                    block_key=key,head=head,target=n,proposal_limit=limits[head][pair]))
                if refresh_mode=='per-head' or head=='approach':
                    events.append(dict(id=key+'_refresh_after_'+head,kind='refresh',cycle=cycle,
                        pair=pair,after_head=head,consumer_heads=list(HEADS)))
        events.append(dict(id=f'cycle{cycle:03d}_finalize',kind='finalize',cycle=cycle))
    return events


def refresh_world_budgets(total, heads=HEADS):
    """Split the existing TOTAL budget, not one full budget per head."""
    if type(total) is not int or total < 0 or not heads:
        raise ValueError('Invalid shared WM refresh budget')
    if total and total < len(heads):
        raise ValueError('A shared refresh must cover each head sampling distribution')
    return {h:total//len(heads)+int(i<total%len(heads)) for i,h in enumerate(heads)}


class AlternatingSchedule:
    schema='supermimic-short-alternating-v1'

    def __init__(self, plan, *, execution_config=None):
        self.plan=copy.deepcopy(plan)
        self.execution_config=copy.deepcopy(execution_config or {})
        self.cursor=0
        self.running=False
        self.results=[]

    @property
    def complete(self):
        return self.cursor==len(self.plan)

    @property
    def current(self):
        return None if self.complete else self.plan[self.cursor]

    def begin(self):
        if self.complete:raise ValueError('Schedule already complete')
        if self.running and self.current['kind']=='refresh':
            raise ValueError('Interrupted refresh: resume pre_refresh checkpoint; partial model/critic updates cannot be replayed')
        self.running=True

    def finish(self, result):
        if not self.running:raise ValueError('No running event')
        event=self.current
        if event['kind']=='train':
            if result.get('accepted')!=event['target']:
                raise ValueError('Short block incomplete; next head/refresh disabled')
            a=result.get('attempts');r=result.get('rejected');b=result.get('backtracked')
            if (type(a) is not int or not event['target']<=a<=event['proposal_limit']
                    or r!=a-event['target'] or type(b) is not int or not 0<=b<=event['target']):
                raise ValueError('Invalid short-block counters')
        elif event['kind']=='refresh':
            if (result.get('complete_episodes')!=128 or result.get('actor_unchanged') is not True
                    or result.get('consumer_heads')!=list(HEADS)):
                raise ValueError('Incomplete combined-policy refresh receipt')
        self.results.append(dict(event=copy.deepcopy(event),result=copy.deepcopy(result)))
        self.cursor+=1
        self.running=False

    def state_dict(self):
        return dict(schema=self.schema,plan=copy.deepcopy(self.plan),execution_config=copy.deepcopy(self.execution_config),cursor=self.cursor,
                    running=self.running,results=copy.deepcopy(self.results))

    def load_state_dict(self, state):
        if (state.get('schema')!=self.schema or state.get('plan')!=self.plan
                or state.get('execution_config')!=self.execution_config):
            raise ValueError('Resume changes alternating schedule/configuration')
        cursor=state.get('cursor')
        if type(cursor) is not int or not 0<=cursor<=len(self.plan) or len(state.get('results',[]))!=cursor:
            raise ValueError('Corrupt alternating cursor/results')
        # Re-validate receipts instead of trusting the stored cursor.
        self.cursor=0;self.running=False;self.results=[]
        for saved in state['results']:
            if saved['event']!=self.current:raise ValueError('Alternating event identity mismatch')
            self.begin();self.finish(saved['result'])
        if type(state.get('running')) is not bool:raise ValueError('Invalid running flag')
        self.running=state['running']
        if self.running and (self.complete or self.current['kind']=='refresh'):
            raise ValueError('Interrupted refresh: use pre_refresh checkpoint; cannot resume partial refresh')

    def head_totals(self, cycle=None):
        totals={h:dict(attempts=0,accepted=0,rejected=0,backtracked=0) for h in HEADS}
        for row in self.results:
            e=row['event']
            if e['kind']=='train' and (cycle is None or e['cycle']==cycle):
                for k in totals[e['head']]:totals[e['head']][k]+=row['result'][k]
        return totals


def run_schedule(schedule, *, train, refresh, finalize, checkpoint):
    """Used by the native entry AND CPU contract tests, with durable boundaries."""
    callbacks=dict(train=train,refresh=refresh,finalize=finalize)
    while not schedule.complete:
        event=copy.deepcopy(schedule.current)
        if event['kind']=='refresh':
            if schedule.running:raise ValueError('Partial refresh cannot be replayed')
            checkpoint('pre_refresh_'+event['id']+'.pt')
            checkpoint('latest.pt')
        schedule.begin()
        result=callbacks[event['kind']](event)
        schedule.finish(result)
        checkpoint('latest.pt')
