"""Check binding against the already checked actual source1 command streams."""
import copy
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
from scripts.initialize_centered_policy_v4 import same_tree


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())


def main():
    path=HERE/'native_factory.py'
    spec=importlib.util.spec_from_file_location('checked_stance_native_factory',path)
    factory=importlib.util.module_from_spec(spec);spec.loader.exec_module(factory)
    check=read(HERE/'CPU_CONTROLLER_CHECK.json')
    assert all(sha(Path(n))==h for n,h in check['source_sha256'].items())
    run=HERE.parent/'early_lateral7_n128_01'
    source_path=run/'candidate_001.pt';data_path=run/'public_and_actions_001.npz';input_path=run/'INPUTS.json'
    source=torch.load(source_path,map_location='cpu',weights_only=False);before=copy.deepcopy(source)
    inputs=read(input_path)
    defaults,limits=[torch.tensor(inputs[k],dtype=torch.float32) for k in ['default_joint_position_rad','joint_limits_rad']]
    with np.load(data_path) as data:public,native,valid=[data[k].copy() for k in ['observations','actions','valid_first_episode']]
    results=[]
    for i in [0,3]:
        candidate=factory.candidates()[i]
        controller=factory.make_controller(source,6,'cpu',**{k:candidate[k] for k in ['gain','horizontal_damping']})
        digest=hashlib.sha256();maximum=0.;previous=None
        for obs,action,keep in zip(public,native,valid):
            if previous is not None:
                ids=torch.tensor(np.flatnonzero(obs[:,775]<previous-1e-6))
                if len(ids):controller.reset(ids)
            value=controller.action({'policy':torch.from_numpy(obs)},defaults,limits)
            digest.update(value.numpy().tobytes())
            if i==0 and keep.any():maximum=max(maximum,float(abs(value.numpy()[keep]-action[keep]).max()))
            previous=obs[:,775].copy()
        earlier=next(x for x in check['results'] if x['stream']=='actual_source1' and x['candidate']['candidate_id']==i)
        assert digest.hexdigest()==earlier['command_stream_sha256']
        assert same_tree(source,before) and maximum<=5e-4
        result=dict(candidate_id=i,actual_public_rows=4800,checked_core_command_stream_bitwise_identical=True,
            command_stream_sha256=digest.hexdigest(),actual_baseline_GPU_max_error=maximum if i==0 else None,
            source_actor_and_normalizer_tree_unchanged=True)
        results.append(result);print(json.dumps(result),flush=True)
    files=[Path(__file__),path,HERE/'stance_controller.py',HERE/'CPU_CONTROLLER_CHECK.json',source_path,data_path,input_path,
        HERE.parent/'windowed_swing_20261027/window_controller.py',HERE.parent/'free_floating_strike_20261027/native_factory.py']
    record=dict(checked_at=datetime.now(timezone.utc).isoformat(),results=results,factory_sha256=sha(path),
        actual_future_momentum_source_branch_pending=True,source_sha256={str(p):sha(p) for p in files},
        new_native_execution=False,formal_acceptance=False,goal_complete=False)
    with (HERE/'NATIVE_FACTORY_CPU_CHECK.json').open('x') as f:f.write(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
