"""Differentiate saved position targets; do not treat this as measured velocity."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
SOURCE = ROOT/'reports/G07_full_racket_v4/normal38_native_tick400_snapshot'
OUT = Path(__file__).parent
CONTRACT = ROOT/'reports/G07_full_racket_v4/CPU_REPLAY_AUTHORED_JOINTS_0724.json'
assert not (OUT/'REPORT.json').exists()
contract = json.loads(CONTRACT.read_text())
names = [f'{leg}_{joint}_joint' for joint in ['hip','thigh','calf'] for leg in ['FL','FR','RL','RR']]
names += [f'joint{i}' for i in range(1,7)]
limits = np.array([contract['limits'][name] for name in names])
with np.load(SOURCE/'control_window.partial.npz') as data:
    action, clock, valid = data['actions'], data['clock_s'], data['valid_first_episode']
    target = limits[:,0]+(action+1)*.5*(limits[:,1]-limits[:,0])
    dt = np.diff(clock,axis=0)
    mask = valid[1:] & valid[:-1] & (dt>0)
    rates = abs(np.diff(target,axis=0)/dt[:,:,None])
    arm = rates[:,:,12:][mask]
    leg = rates[:,:,:12][mask]
with np.load(SOURCE/'near_contact_all.partial.npz') as data:
    actual = abs(data['observations'][:,:,27:45])[data['valid_first_episode']]
quantiles = lambda array: {str(q):float(np.quantile(array,q)) for q in [0,.5,.9,.95,.99,1.]}
paths = [SOURCE/'control_window.partial.npz',SOURCE/'near_contact_all.partial.npz',CONTRACT,Path(__file__)]
report = dict(created_at=datetime.now(timezone.utc).isoformat(),
    scope='Saved CEM21generation0actual public/control window, valid first episodes only. Differentiated commanded position is an implied command rate, not measured velocity or an applied physical force.',
    native_window_ticks=int(len(action)), commanded_arm_rate_quantiles_rad_s=quantiles(arm),
    commanded_leg_rate_quantiles_rad_s=quantiles(leg),
    commanded_arm_rate_above5_05_fraction=float(np.mean(arm>5.05)),
    command_rate_diagnostic_threshold_rad_s=5.05,
    measured_arm_velocity_quantiles_rad_s=quantiles(actual[:,12:]),
    configured_nominal_arm_velocity_limit_rad_s=5.,
    observed_measured_arm_overshoot_above_nominal_rad_s=float(actual[:,12:].max()-5.),
    overshoot_retained_without_clipping_observations=True,
    note='The5.05command diagnostic threshold includes0.05for differentiation roundoff; it never changes the configured5rad/s limit or acceptance. Actual5.37rad/s solver overshoot remains visible. A future software trajectory rate bound would need new native contact/recovery validation; none is claimed here.',
    proposed_followup='Evaluate smooth bounded joint targets with anticipatory timing while preserving original actuator limits and all task gates.',
    new_controller_executed=False,formal_acceptance=False,
    sha256={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
(OUT/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='sha256'},indent=2))
