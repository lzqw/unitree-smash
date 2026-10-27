"""Check source dispatch against two actual saved native input/action streams."""
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


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    factory=load('checked_momentum_native_factory',HERE/'native_factory.py')
    core=factory.checked_core()
    assert factory.candidates()==core.candidates()
    run=HERE.parent/'early_lateral7_n128_01'
    inputs=json.loads((run/'INPUTS.json').read_text())
    defaults,limits=[torch.tensor(inputs[k],dtype=torch.float32) for k in
                     ['default_joint_position_rad','joint_limits_rad']]
    files=[Path(__file__),HERE/'native_factory.py',HERE/'strike_controller.py',HERE/'free_floating.py',
        HERE/'CPU_CONTROLLER_CHECK.json',HERE/'ACTUAL_CHECK.json',run/'INPUTS.json',
        HERE.parent/'windowed_swing_20261027/window_controller.py',
        HERE.parent/'public_air_countermotion_20261027/native_factory.py']
    results=[]
    for suffix in ['000','001']:
        path,data_path=run/f'candidate_{suffix}.pt',run/f'public_and_actions_{suffix}.npz'
        source=torch.load(path,map_location='cpu',weights_only=False)
        saved=copy.deepcopy(source)
        with np.load(data_path) as data:
            public,actions,valid=[data[k].copy() for k in ['observations','actions','valid_first_episode']]
            assert data['training_indices'].tolist()==[0,29,69,100,112,123]
        assert public.shape==(800,6,794)
        configs=[factory.candidates()[0]]
        if suffix=='001':configs.append(factory.candidates()[3])
        for candidate in configs:
            kwargs={k:candidate[k] for k in ['gain','leg_velocity_delta_cap_rad_s']}
            bound=factory.make_controller(source,6,'cpu',**kwargs)
            direct=core.MomentumStrikeController(factory.make_source(source,6,'cpu'),source['asset_sha256'],6,'cpu',**kwargs)
            maximum=0.;previous=None;digest=hashlib.sha256()
            for obs,native,keep in zip(public,actions,valid):
                if previous is not None:
                    ids=torch.tensor(np.flatnonzero(obs[:,775]<previous-1e-6))
                    if len(ids):bound.reset(ids);direct.reset(ids)
                value=torch.from_numpy(obs)
                a=bound.action({'policy':value},defaults,limits)
                b=direct.action({'policy':value},defaults,limits)
                assert torch.equal(a,b)
                digest.update(a.numpy().tobytes())
                if not candidate['gain'] and keep.any():
                    maximum=max(maximum,float(abs(a.numpy()[keep]-native[keep]).max()))
                previous=obs[:,775].copy()
            assert maximum<=5e-4 and same_tree(source,saved)
            result=dict(source_candidate=int(suffix),candidate_id=candidate['candidate_id'],actual_public_rows=4800,
                valid_first_episode_rows=int(valid.sum()),native_factory_matches_checked_core_bitwise=True,
                actual_baseline_CPU_vs_GPU_max_error=(maximum if not candidate['gain'] else None),
                command_stream_sha256=digest.hexdigest(),source_weights_unchanged=True)
            results.append(result);print(json.dumps(result),flush=True)
        files.extend([path,data_path])
    record=dict(checked_at=datetime.now(timezone.utc).isoformat(),results=results,
        factory_sha256=sha(HERE/'native_factory.py'),
        actual_baseline_CPU_vs_GPU_max_error=max(x['actual_baseline_CPU_vs_GPU_max_error'] or 0. for x in results),
        actual_future_countermotion_source_branch_pending=True,
        source_sha256={str(p):sha(p) for p in files},native_execution=False,formal_acceptance=False)
    with (HERE/'NATIVE_FACTORY_CPU_CHECK.json').open('x') as stream:stream.write(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
