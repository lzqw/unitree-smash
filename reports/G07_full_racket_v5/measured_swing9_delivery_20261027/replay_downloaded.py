"""Check delivered actual search/cold records and replay their real public inputs."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--packet',type=Path,required=True)
parser.add_argument('--runtime',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
packet,runtime=args.packet.resolve(),args.runtime.resolve()
assert packet.is_dir() and runtime.is_dir()
original_reads=[]


def audit(event,values):
    if event=='open' and values and isinstance(values[0],(str,bytes,os.PathLike)):
        path=os.fsdecode(values[0])
        if path.startswith('/home/autodrive/LZQW/unitree_smash') and path.endswith(('.py','.pyc')):
            original_reads.append(path)
            raise RuntimeError('Original live project Python access forbidden: '+path)


sys.addaudithook(audit)
sys.path=[str(runtime)]+[p for p in sys.path if not str(p).startswith('/home/autodrive/LZQW/unitree_smash')]
os.chdir(runtime)
import numpy as np
import torch
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases,summarize_training
from quadsmash.control.full_racket_body_search_v4 import summarize_candidate
from scripts.initialize_centered_policy_v4 import same_tree


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def main():
    manifest=read(packet/'MANIFEST.json')
    for name,row in manifest['payloads'].items():
        p=packet/name
        assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],name
    def resolve(name):return packet/manifest['aliases'].get(name,name)
    source=torch.load(resolve('source/original204.pt'),map_location='cpu',weights_only=False)
    factory_path=resolve('search/extra_execution_sources/swing_controller.py')
    spec=importlib.util.spec_from_file_location('downloaded_actual_swing_factory',factory_path)
    factory=importlib.util.module_from_spec(spec);spec.loader.exec_module(factory)
    protocol=read(runtime/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json');protocol['_root']=str(runtime)
    cases,indices=training_cases(runtime,protocol)
    assert indices==list(range(128))
    results=[];counts=[]
    for label in ['search','cold128']:
        hashes=read(resolve(label+'/execution_source_hashes.json'))
        assert len(hashes)==394 and all(sha(runtime/name)==digest for name,digest in hashes.items())
        inputs=read(resolve(label+'/INPUTS.json'));metrics=read(resolve(label+'/metrics.json'))
        defaults,limits=[torch.tensor(inputs[n],dtype=torch.float32) for n in ['default_joint_position_rad','joint_limits_rad']]
        for entry in metrics['history']:
            i=entry['candidate_id'];suffix=f'{i:03d}'
            rows=read(resolve(label+f'/episodes_{suffix}.json'))
            actual=summarize_candidate(rows,protocol)
            assert all(actual[k]==entry['summary'][k] for k in actual)
            if label=='cold128':
                cold=summarize_training(rows,cases,indices,protocol)
                assert all(cold[k]==entry['summary'][k] for k in ['complete','airborne_contacts','conditional_recovery_count','numerical_failures'])
                policy_path=resolve('search/best_candidate.pt')
            else:policy_path=resolve(f'search/candidate_{suffix}.pt')
            policy=torch.load(policy_path,map_location='cpu',weights_only=False)
            assert same_tree(policy['source_public_policy'],source)
            assert policy['factory_sha256']==sha(factory_path)
            assert torch.equal(policy['parameters'],source['parameters'])
            controller=factory.MeasuredSwingController(source,6,'cpu',policy['runtime_overrides'])
            with np.load(resolve(label+f'/public_and_actions_{suffix}.npz')) as data:
                public,actions,valid=[data[n].copy() for n in ['observations','actions','valid_first_episode']]
                assert data['training_indices'].tolist()==[0,29,69,100,112,123]
            previous=None;maximum=0.
            with torch.inference_mode():
                for values,native,keep in zip(public,actions,valid):
                    obs=torch.from_numpy(values)
                    if previous is not None:
                        reset=torch.where(obs[:,775]<previous-1e-5)[0]
                        if len(reset):controller.reset(reset)
                    result=controller.action({'policy':obs},defaults,limits)
                    assert result.shape==(6,18) and torch.isfinite(result).all() and (abs(result)<=1).all()
                    if keep.any():maximum=max(maximum,float(abs(result.numpy()[keep]-native[keep]).max()))
                    previous=obs[:,775].clone()
            assert maximum<=5e-4,(label,i,maximum)
            row=dict(run=label,candidate_id=i,actual_valid_public_rows=int(valid.sum()),
                CPU_vs_actual_GPU_max_action_error=maximum,actual_original128_summary=actual)
            results.append(row);counts.append(len(rows))
            print(json.dumps(dict(run=label,candidate_id=i,actual_valid_public_rows=int(valid.sum()),maximum_error=maximum)),flush=True)
    assert len(results)==10 and sum(counts)==1280 and not original_reads
    record=dict(checked_at=datetime.now(timezone.utc).isoformat(),packet=str(packet),restored_runtime=str(runtime),
        actual_raw_episodes_recomputed=1280,actual_execution_sources_per_run_verified=394,
        results=results,original_action_tolerance=5e-4,
        original_live_project_Python_reads_rejected=True,observed_original_project_Python_reads=original_reads,
        manifest_sha256=sha(packet/'MANIFEST.json'),program_sha256=sha(Path(__file__)),
        new_native_physics_reexecution=False,formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    with args.output.open('x') as stream:stream.write(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
