"""Exercise the separate factory baseline on real original native inputs."""
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
MAIN=HERE.parents[2]
ROOT=MAIN.parent/'unitree_smash_contact_recovery'
sys.path.insert(0,str(ROOT))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    spec=importlib.util.spec_from_file_location('checked_native_air_factory',HERE/'native_factory.py')
    factory=importlib.util.module_from_spec(spec);spec.loader.exec_module(factory)
    cpu=json.loads((HERE/'CPU_CONTROLLER_CHECK.json').read_text())
    assert all(sha(Path(p))==h for p,h in cpu['source_sha256'].items())
    measured=HERE.parent/'measured_swing9_n128_01'
    policy=MAIN/'reports/G07_full_racket_v4/gain_body8_sequential_n128_01/best_candidate.pt'
    source=torch.load(policy,map_location='cpu',weights_only=False)
    inputs=json.loads((measured/'INPUTS.json').read_text())
    defaults,limits=[torch.tensor(inputs[n],dtype=torch.float32) for n in ['default_joint_position_rad','joint_limits_rad']]
    with np.load(measured/'public_and_actions_000.npz') as data:
        public,actions=data['observations'].copy(),data['actions'].copy()
    controller=factory.make_controller(source,6,'cpu',0.)
    maximum=0.
    for value,native in zip(public,actions):
        actual=controller.action({'policy':torch.from_numpy(value)},defaults,limits)
        maximum=max(maximum,float(abs(actual.numpy()-native).max()))
    assert maximum<=5e-4
    assert [c['maximum_velocity_change_rad_s'] for c in factory.candidates()]==[0.,2.,5.]
    record=dict(checked_at=datetime.now(timezone.utc).isoformat(),factory_sha256=sha(HERE/'native_factory.py'),
        core_all3_CPU_check_sha256=sha(HERE/'CPU_CONTROLLER_CHECK.json'),
        actual_baseline_factory_public_rows=4800,actual_baseline_CPU_vs_GPU_max_error=maximum,
        actual_recovery_prior_source_factory_branch_pending_completed_source=True,
        source_sha256={str(p):sha(p) for p in [Path(__file__),HERE/'native_factory.py',HERE/'countermotion.py',
            HERE/'CPU_CONTROLLER_CHECK.json',policy,measured/'public_and_actions_000.npz',measured/'INPUTS.json']},
        native_execution=False,formal_acceptance=False)
    with (HERE/'NATIVE_FACTORY_CPU_CHECK.json').open('x') as stream:stream.write(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(actual_baseline_factory_public_rows=4800,maximum_baseline_error=maximum,
        actual_future_source_checks_pending=True)))


if __name__=='__main__':main()
