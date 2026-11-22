"""Stop-only monitoring of complete frozen training episodes; never promotion."""
import math


class FrozenReturnMonitor:
    schema='complete-frozen-native-return-trend-v1'
    def __init__(self,drop_tolerance=.1,flat_tolerance=.05,patience=2):
        if any(not math.isfinite(x) or x<0 for x in (drop_tolerance,flat_tolerance)) or type(patience)is not int or patience<1:
            raise ValueError('Invalid native return monitoring thresholds')
        self.config=dict(drop_tolerance=drop_tolerance,flat_tolerance=flat_tolerance,patience=patience)
        self.rows=[];self.decline=0;self.flat=0;self.stop_reason=None

    def observe(self,records,*,policy_id,actor_sha256,contract_sha256,stop_on_plateau=True):
        if self.stop_reason:raise RuntimeError('Native return monitor already stopped: '+self.stop_reason)
        if not policy_id or not actor_sha256 or not contract_sha256:raise ValueError('Missing frozen policy/contract identity')
        if len(records)!=128 or {r['env_id'] for r in records}!=set(range(128)):
            raise ValueError('Native return monitoring requires all128 distinct lanes')
        if len({r['case_id'] for r in records})!=128:raise ValueError('Duplicate/missing case identity')
        if any(r.get('policy_id')!=policy_id or r.get('actor_sha256')!=actor_sha256 or r.get('mixed_policy') is True for r in records):
            raise ValueError('Mixed or mismatched frozen policy labels')
        if any(r.get('native_return_definition')!='undiscounted_sum_of_native_control_step_rewards_v1'
               or not 1<=r.get('native_episode_length_steps',0)<=800 for r in records):
            raise ValueError('Incomplete or incompatible native episode returns')
        returns=[r['native_episode_return'] for r in records]
        if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in returns):raise ValueError('Nonfinite native return')
        mean=sum(returns)/128
        row=dict(policy_id=policy_id,actor_sha256=actor_sha256,contract_sha256=contract_sha256,
                 mean=mean,episodes=128,stop_on_plateau=stop_on_plateau,case_ids=sorted(r['case_id'] for r in records))
        if self.rows:
            previous=self.rows[-1]
            if previous['contract_sha256']!=contract_sha256 or previous['case_ids']!=row['case_ids']:
                raise ValueError('Native monitoring protocol/case set changed')
            delta=mean-previous['mean'];row['delta']=delta
            self.decline=self.decline+1 if delta < -self.config['drop_tolerance'] else 0
            repeated=previous['actor_sha256']==actor_sha256
            row['same_actor_as_previous']=repeated
            if not repeated:
                self.flat=self.flat+1 if abs(delta)<=self.config['flat_tolerance'] else 0
            if self.decline>=self.config['patience']:self.stop_reason='native-frozen-return-declining'
            elif stop_on_plateau and self.flat>=self.config['patience']:self.stop_reason='native-frozen-return-plateau'
        self.rows.append(row)
        return dict(row,decline_streak=self.decline,flat_streak=self.flat,stop_reason=self.stop_reason,
                    config=self.config,formal_promotion=False,
                    scope='Complete frozen training stream trend; plateau is not proof of policy optimality')

    def state_dict(self):
        return dict(schema=self.schema,config=self.config,rows=list(self.rows),decline=self.decline,flat=self.flat,stop_reason=self.stop_reason)

    def load_state_dict(self,state):
        if state['schema']!=self.schema or state['config']!=self.config:raise ValueError('Native monitor resume contract differs')
        self.rows=list(state['rows']);self.decline=int(state['decline']);self.flat=int(state['flat']);self.stop_reason=state['stop_reason']
        if not self.stop_reason and any(a['actor_sha256']==b['actor_sha256'] for a,b in zip(self.rows,self.rows[1:])):
            # Older runs counted re-evaluation of one frozen policy as another
            # optimizer result. Reconstruct only the plateau streak on resume.
            self.flat=0
            for a,b in zip(self.rows,self.rows[1:]):
                if a['actor_sha256']!=b['actor_sha256']:
                    self.flat=self.flat+1 if abs(b['mean']-a['mean'])<=self.config['flat_tolerance'] else 0
