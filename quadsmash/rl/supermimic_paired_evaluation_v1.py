"""Paired fixed-case evaluation for SuperMimic update audits."""

from collections import Counter
import math
import torch

from .supermimic_native_metrics import summarize_native_episodes


def _case_key(row):
    if 'case_id' not in row:
        raise ValueError('Paired evaluation requires case_id')
    return str(row['case_id'])


def verify_baseline(rows, actor_hash, execution):
    if len(rows) != 128 or len({_case_key(row) for row in rows}) != 128:
        raise ValueError('Expected exactly 128 unique baseline cases')
    if any(row.get('actor_sha256') != actor_hash for row in rows):
        raise ValueError('Baseline differs from loaded actor or lacks actor identity')
    if any(row.get('policy_execution') != execution for row in rows):
        raise ValueError('Baseline policy execution does not match this run')


def failure_taxonomy(row):
    """Return mutually useful failure labels without dropping successes."""
    labels=[]
    if row.get('numerical_failures', 0): labels.append('numerical_failure')
    if row.get('rally_fault', False): labels.append('rally_fault')
    if not row.get('qualified_airborne_string_contact', False): labels.append('no_qualified_contact')
    if row.get('qualified_airborne_string_contact', False) and not row.get('ball_side_complete_except_recovery', False):
        labels.append('ball_side_failure')
    if row.get('ball_side_complete_except_recovery', False) and not row.get('recovery_success', False):
        labels.append('recovery_failure')
    if not row.get('complete5', False): labels.append('complete5_failure')
    if row.get('body_ground_seen', False): labels.append('body_ground')
    return labels or ['success']


def _taxonomy(rows):
    counts=Counter()
    for row in rows: counts.update(failure_taxonomy(row))
    return dict(sorted(counts.items()))


def native_critic_td(values, returns, identities, gamma):
    """One-step Bellman residual on consecutive states within a frozen native episode."""
    values = torch.as_tensor(values).detach().cpu().reshape(-1).double()
    returns = torch.as_tensor(returns).detach().cpu().reshape(-1).double()
    if len(values) != len(identities) or len(returns) != len(values):
        raise ValueError('Critic audit requires aligned state, RTG and value rows')
    by_state = {}
    for i, identity in enumerate(identities):
        key = tuple(map(int, identity))
        if key in by_state:
            prior = by_state[key]
            if not (torch.isclose(values[prior], values[i]) and torch.isclose(returns[prior], returns[i])):
                raise ValueError('Conflicting duplicate frozen state')
        by_state[key] = i
    pairs = [(i, by_state[(lane, ep, t+1)]) for (lane, ep, t), i in by_state.items()
             if (lane, ep, t+1) in by_state]
    if not pairs:
        return dict(sample_count=0, available=False)
    current, following = torch.tensor(pairs).T
    reward = returns[current] - gamma * returns[following]
    residual = reward + gamma * values[following] - values[current]
    return dict(sample_count=len(pairs), available=True, gamma=gamma,
                mae=float(residual.abs().mean()), rmse=float(residual.square().mean().sqrt()),
                bias=float(residual.mean()), scope='adjacent observed frozen states; reward recovered from native RTG')


def parameter_delta(before, after):
    if before is None or after is None:
        return {'available': False}
    keys=sorted(set(before).intersection(after))
    if not keys: raise ValueError('No common actor parameters for paired delta')
    values=[]
    for key in keys:
        a,b=before[key],after[key]
        if torch.is_tensor(a) and torch.is_tensor(b) and a.shape==b.shape:
            values.append((b.detach().float()-a.detach().float()).reshape(-1))
    if not values:return {'available': False}
    vector=torch.cat(values)
    return {'available': True,'l2':float(vector.norm()),'rms':float(vector.square().mean().sqrt()),
            'max_abs':float(vector.abs().max()),'changed_elements':int((vector!=0).sum())}


def paired_flips(rows, field):
    improved = [row['case_id'] for row in rows if row[field+'_after'] and not row[field+'_before']]
    regressed = [row['case_id'] for row in rows if row[field+'_before'] and not row[field+'_after']]
    discordant = len(improved) + len(regressed)
    tail = sum(math.comb(discordant, i) for i in range(min(len(improved), len(regressed)) + 1))
    return dict(improved_cases=improved, regressed_cases=regressed,
                paired_two_sided_p=min(1., 2 * tail / 2**discordant))


def paired_evaluation(pre, post, *, stage, actor_before=None, actor_after=None,
                      action_before=None, action_after=None, critic_td=None,
                      wm_holdout=None):
    """Compare two records for exactly the same fixed case identities."""
    pre_map={_case_key(row):row for row in pre};post_map={_case_key(row):row for row in post}
    if len(pre)!=128 or len(post)!=128 or len(pre_map)!=128 or len(post_map)!=128 or set(pre_map)!=set(post_map):
        raise ValueError('Paired evaluation requires the same 128 unique case ids')
    before=[pre_map[key] for key in sorted(pre_map)]
    after=[post_map[key] for key in sorted(post_map)]
    before_metrics=summarize_native_episodes(before)
    after_metrics=summarize_native_episodes(after)
    deltas={key: after_metrics.get(key)-before_metrics.get(key)
            for key in ('qualified_airborne_string_contact_rate','ball_side_complete_except_recovery_rate',
                        'recovery_given_airborne_string_contact','complete5_rate','native_episode_return_mean')
            if isinstance(after_metrics.get(key),(int,float)) and isinstance(before_metrics.get(key),(int,float))}
    case_deltas=[]
    for key in sorted(pre_map):
        a,b=pre_map[key],post_map[key]
        case_deltas.append(dict(case_id=key,
            return_delta=float(b['native_episode_return'])-float(a['native_episode_return']),
            contact_before=bool(a.get('qualified_airborne_string_contact',False)),
            contact_after=bool(b.get('qualified_airborne_string_contact',False)),
            recovered_before=bool(a['recovery_success']),recovered_after=bool(b['recovery_success']),
            ball_side_before=bool(a.get('ball_side_complete_except_recovery',False)),
            ball_side_after=bool(b.get('ball_side_complete_except_recovery',False)),
            complete5_before=bool(a.get('complete5',False)),complete5_after=bool(b.get('complete5',False)),
            failure_before=failure_taxonomy(a),failure_after=failure_taxonomy(b)))
    common=[row for row in case_deltas if row['contact_before'] and row['contact_after']]
    conditional=dict(common_contacts=len(common), **paired_flips(common,'recovered'),
                     recovered_before=sum(row['recovered_before'] for row in common),
                     recovered_after=sum(row['recovered_after'] for row in common),
                     gained_contacts=[r['case_id'] for r in case_deltas if r['contact_after'] and not r['contact_before']],
                     lost_contacts=[r['case_id'] for r in case_deltas if r['contact_before'] and not r['contact_after']])
    result=dict(paired_recovery=conditional,schema='supermimic-paired-fixed-case-evaluation-v1',stage=stage,
        case_count=128,case_ids=sorted(pre_map),before=before_metrics,after=after_metrics,
        deltas=deltas,failure_taxonomy_before=_taxonomy(before),failure_taxonomy_after=_taxonomy(after),
        case_deltas=case_deltas,actor_parameter_delta=parameter_delta(actor_before,actor_after),
        actor_kl=None,action_rms=None,critic_td=critic_td,wm_holdout=wm_holdout,
        action_metrics_available=action_before is not None and action_after is not None,
        paired_case_identity_verified=True)
    if action_before is not None and action_after is not None:
        if isinstance(action_before,dict):
            a=torch.as_tensor(action_before['action']).float()
        else:a=torch.as_tensor(action_before).float()
        if isinstance(action_after,dict):
            b=torch.as_tensor(action_after['action']).float()
        else:b=torch.as_tensor(action_after).float()
        if a.shape!=b.shape:raise ValueError('Paired action tensors must have equal shapes')
        delta=b-a
        result['action_rms']=float(delta.square().mean().sqrt())
        if (isinstance(action_before,dict) and isinstance(action_after,dict)
                and action_before.get('log_std') is not None and action_after.get('log_std') is not None):
            if ((action_before.get('clipped_action') and (a.abs() >= 1).any()) or
                    (action_after.get('clipped_action') and (b.abs() >= 1).any())):
                result['actor_kl_available'] = False
                result['actor_kl_reason'] = 'clipped means cannot identify Gaussian KL'
                return result
            mu0,mu1=a,b
            log0=torch.as_tensor(action_before['log_std']).float().reshape(1,-1)
            log1=torch.as_tensor(action_after['log_std']).float().reshape(1,-1)
            var0, var1=(2.*log0).exp(),(2.*log1).exp()
            result['actor_kl']=float(.5*(2*(log1-log0)+(var0+(mu0-mu1).square())/var1-1.).mean())
            result['actor_kl_available']=True
            result['actor_kl_scope']='pre-clamp diagonal Gaussian KL, mean per action dimension'
        else:
            result['actor_kl_available']=False
    else:
        result['actor_kl_available']=False
    return result
