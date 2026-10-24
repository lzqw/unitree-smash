"""Independent count/legality checks for frozen full-racket evaluation panels."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .physics.rally_rules_v4 import RallyRulesV4

FAULTS=tuple(name for name in RallyRulesV4.flags if name.endswith('_fault') or name=='engineering_body_crossing')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_frozen_inputs(root,protocol_path,lock_path,checkpoint_path):
    root=Path(root);protocol=json.loads(Path(protocol_path).read_text());lock=json.loads(Path(lock_path).read_text())
    if lock['schema']!='full-racket-policy-lock-v4':raise ValueError('Unsupported policy lock')
    if digest(protocol_path)!=lock['protocol_sha256'] or digest(checkpoint_path)!=lock['policy_sha256']:
        raise ValueError('Policy or protocol changed after lock')
    if digest(root/protocol['asset']['path'])!=protocol['asset']['sha256'] or lock['asset_sha256']!=protocol['asset']['sha256']:
        raise ValueError('Native asset differs from the frozen protocol')
    required={str(path.relative_to(root)) for path in root.glob('quadsmash/**/*.py')}
    required.update({'scripts/evaluate_full_racket_v4.py','scripts/lock_full_racket_policy_v4.py','scripts/aggregate_full_racket_eval_v4.py'})
    if not required.issubset(lock['source_hashes']):raise ValueError('Policy lock omits evaluation or control source')
    for path,expected in {**protocol['physical_configs'],**lock['source_hashes']}.items():
        if digest(root/path)!=expected:raise ValueError('Frozen dependency changed: '+path)
    for suite in protocol['formal_suites'].values():
        if suite['count']!=200 or digest(root/suite['path'])!=suite['sha256']:
            raise ValueError('Frozen200-case panel changed')
    expected=dict(complete_success_rate=.5,airborne_contact_rate=.7,conditional_recovery_rate=.85,numerical_failures=0)
    if protocol['thresholds']!=expected:raise ValueError('Original G07 thresholds must be retained')
    return protocol,lock


def independent_episode(row,protocol):
    for name in FAULTS:
        if name not in row:raise ValueError('Missing legality flag: '+name)
    fault=any(row[name] for name in FAULTS)
    contact=np.asarray(row['contact_point_m'],dtype=float)
    landing=np.asarray(row['first_landing_m'],dtype=float)
    velocity=np.asarray(row['outgoing_velocity_mps'],dtype=float)
    court=json.loads(Path(protocol['_root'],'configs/court_net_v4.json').read_text())
    legal=(row['legal_initial_contact'] and np.isfinite(contact).all() and contact[0]<court['net_plane_x_m'])
    air=bool(row['airborne_string_hit'] and row['string_hit'] and legal)
    jump=row['jump']
    recovery_protocol=json.loads(Path(protocol['_root'],'configs/evaluation_protocol_v2.json').read_text())['recovery']
    recovered=bool(jump['qualified_jump_landed'] and jump['stable_recovery'] and not jump['body_landing']
        and not jump['terminated'] and jump['recovery_s']>=recovery_protocol['continuous_stable_window_s']-1e-6
        and jump['maximum_body_ground_force_n']<=recovery_protocol['maximum_body_ground_force_n'])
    target=(np.isfinite(landing).all() and court['target_x_limits_m'][0]<=landing[0]<=court['target_x_limits_m'][1]
        and abs(landing[1])<=court['target_y_half_width_m']
        and court['opponent_x_limits_m'][0]<=landing[0]<=court['opponent_x_limits_m'][1]
        and abs(landing[1])<=court['singles_width_m']/2)
    tolerance=protocol['physics_dt_s']
    ordered=(np.isfinite([row['contact_time_s'],row['release_time_s'],row['crossing_time_s'],row['first_landing_time_s']]).all()
        and row['contact_time_s']>=protocol['preparation_seconds']-tolerance
        and row['release_time_s']>=row['contact_time_s']
        and row['crossing_time_s']+tolerance>=row['contact_time_s']
        and row['first_landing_time_s']+tolerance>=max(row['crossing_time_s'],row['release_time_s']))
    attack=(air and row['released'] and np.isfinite(velocity).all() and np.linalg.norm(velocity)>=8.
            and velocity[0]>0 and velocity[2]<=-1.)
    complete=bool(row['ground_init'] and row['active_rally_launch_count']==1 and attack
        and row['crossed_net'] and row['landed_target'] and target and ordered and recovered and not fault)
    if bool(row['complete_smash'])!=complete:raise ValueError('Native/independent complete flag disagreement in case '+row['case_id'])
    return dict(airborne_contact=air,conditional_recovery=bool(air and recovered and not fault),
                complete=complete,fault=fault)


def summarize_panel(rows,protocol,panel):
    suite=protocol['formal_suites'][panel]
    cases=json.loads(Path(protocol['_root'],suite['path']).read_text())['cases']
    if len(rows)!=200 or [row['case_id'] for row in rows]!=[case['case_id'] for case in cases]:
        raise ValueError('Every frozen case must appear exactly once in its original order')
    if any(r['episode_index']!=0 or r['env_id']!=i for i,r in enumerate(rows)):
        raise ValueError('Evaluation must use the first episode of each assigned case')
    if any(not r['ground_init'] or r['active_rally_launch_count']>1 for r in rows):
        raise ValueError('Ground reset or single active launch contract violated')
    if any(not isinstance(r['numerical_failures'],int) or r['numerical_failures']<0 for r in rows):
        raise ValueError('Numerical failure counts must be nonnegative integers')
    checked=[independent_episode(row,protocol) for row in rows]
    air=sum(r['airborne_contact'] for r in checked)
    recovered=sum(r['conditional_recovery'] for r in checked)
    complete=sum(r['complete'] for r in checked)
    numerical=sum(r['numerical_failures'] for r in rows)
    return dict(panel=panel,episodes=200,complete=complete,complete_success_rate=complete/200,
        airborne_contacts=air,airborne_contact_rate=air/200,conditional_recovery_count=recovered,
        conditional_recovery_rate=recovered/air if air else 0.,numerical_failures=numerical,
        actual_racket_contacts=sum(r['had_contact'] for r in rows),
        crossed_net=sum(r['crossed_net'] for r in rows),landed_legal_court=sum(r['landed_legal_court'] for r in rows),
        landed_target=sum(r['landed_target'] for r in rows),
        fault_counts={name:sum(r[name] for r in rows) for name in FAULTS},
        fixed_case_ids=[r['case_id'] for r in rows],formal_eligible=True)


def aggregate_panels(panels,protocol):
    if {p['panel'] for p in panels}!={'A','B'} or len(panels)!=2:
        raise ValueError('Exactly the two distinct frozen panels are required')
    for key in ('policy_sha256','protocol_sha256','policy_lock_sha256'):
        if len({p[key] for p in panels})!=1:raise ValueError('Panels used different '+key)
    if any(p['episodes']!=200 or not p['formal_eligible'] for p in panels):
        raise ValueError('Each panel must contain200 eligible cases')
    rates={key:min(p[key] for p in panels) for key in
        ('complete_success_rate','airborne_contact_rate','conditional_recovery_rate')}
    numerical=sum(p['numerical_failures'] for p in panels)
    checks={key:rates[key]>=protocol['thresholds'][key] for key in rates}
    checks['numerical_failures']=numerical==0
    return dict(sample_count=400,panels=panels,worst_panel_rates=rates,
        numerical_failures=numerical,checks=checks,formal_acceptance=all(checks.values()),
        policy_sha256=panels[0]['policy_sha256'],protocol_sha256=panels[0]['protocol_sha256'],
        scope='Frozen source/weight simulation policy thresholds only; full goal deliverables remain separately auditable.')
