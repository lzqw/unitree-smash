"""Validate/seal all four real frozen TRAIN128 assessments; CPU only."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
PREFIX='reports/G07_full_racket_v4'
TRAIN=PREFIX+'/body_residual_leg15_fresh_n2048_updates48_01'
OWNER=PREFIX+'/body_residual48_after_grid16_queue01'
COMPARE=PREFIX+'/body48_fixed_case_comparison_20261026'
ITERATIONS=[13,24,36,48]
ARCHIVE=ROOT/PREFIX/'G07_body48_frozen13_24_36_48_full_TRAIN128_weights_and_records_20261027.zip'
LAYERS=[
    ('G07_placement075_stage8_to40_completed_native_and_frozen_TRAIN128_20261026.zip','44528da938ad422ed6a5515a32c65df3b763421288d2b4ff6f9236ab36c517bb'),
    ('G07_body_residual48_completed_native_training_and_all_weights_20261026.zip','8dbe30b677522e86176f040520d5ced9ccc6acc507b2b02bbfd101681ade40ee'),
]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return digest_stream(stream)[1]


def digest_stream(stream):
    size,result=0,hashlib.sha256()
    for block in iter(lambda:stream.read(8*1024*1024),b''):
        result.update(block)
        size+=len(block)
    return size,result.hexdigest()


def save(path,value):
    assert not path.exists(),str(path)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def policy_path(root,iteration):
    return root/TRAIN/'policy.pt' if iteration==48 else root/OWNER/f'exports/iter{iteration:02d}/policy.pt'


def check(root,require_bundled=False):
    root=root.resolve()
    sys.path[:0]=[str(root/'vendor/rsl_rl'),str(root)]
    import numpy as np
    import torch
    import rsl_rl
    from tensordict import TensorDict
    from quadsmash.control.full_racket_body_residual_v4 import make_rally_controller
    from scripts.initialize_centered_policy_v4 import same_tree
    bundled=Path(rsl_rl.__file__).resolve().is_relative_to(root/'vendor/rsl_rl')
    assert bundled or not require_bundled
    compare=module('actual_frozen_four_case_checks',root/COMPARE/'compare.py')
    helper=module('actual_frozen_four_actor_loader',root/OWNER/'export_intermediate.py')
    protocol=read(root/'configs/g07_full_racket_eval_v4_r3.json')
    protocol['_root']=str(root)
    config=read(root/'configs/g07_full_racket_body_residual_leg15_v4.json')
    source_rel=PREFIX+'/body_strike_grid16_n2048_01/best_candidate.pt'
    base=torch.load(root/source_rel,map_location='cpu',weights_only=False)
    provenance=read(root/TRAIN/'trajectory_traces/manifest.json')['provenance']
    default,limits=[torch.tensor(provenance[key],dtype=torch.float32)
        for key in ['default_joint_position_rad','joint_limits_rad']]
    original_hashes=read(root/TRAIN/'source_hashes.json')
    for name,digest in original_hashes.items():
        assert sha(root/name)==digest,name
    results=[]
    for iteration in ITERATIONS:
        run=root/PREFIX/f'body_residual_fresh_iter{iteration:02d}_fixed_train128_01'
        policy=policy_path(root,iteration)
        rows,summary=compare.load(run,policy,protocol)
        assert read(run/'source_hashes.json')==original_hashes
        initial=read(run/'initial_states.json')
        assert [row['env_id'] for row in initial]==list(range(128))
        assert np.max(abs(np.asarray([row['joint_position_rad'] for row in initial])-default.numpy()))<1e-5
        assert np.max(abs(np.asarray([row['joint_velocity_rad_s'] for row in initial])))<1e-5
        with np.load(run/'all_first_episodes.npz') as arrays:
            valid=arrays['valid_first_episode'].astype(bool)
            assert valid.shape==(400,128) and valid[0].all()
            assert arrays['link_poses'].shape==(400,128,28,7)
            assert np.isfinite(arrays['link_poses'][valid]).all()
            assert np.isfinite(arrays['ball_position_m'][valid]).all()
            assert abs(arrays['episode_clock_s'][0]).max()<1e-6
        public=torch.load(policy,map_location='cpu',weights_only=False)
        assert public['schema']=='full-racket-body-residual-ppo-v4' and public['native_execution']
        assert not public['performance_inherited'] and same_tree(public['base_controller'],base)
        checkpoint=root/TRAIN/('checkpoint.pt' if iteration==48 else f'ppo/model_{iteration-1}.pt')
        native=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert native['iter']==iteration
        assert bool(native['native']['canonical_restart'])==(iteration==48)
        assert native['native']['total_timesteps']==iteration*262144
        for name in ['actor_obs_normalizer.count','critic_obs_normalizer.count']:
            assert int(native['model_state_dict'][name])==iteration*262144
        actor=helper.from_state(native,config)
        with np.load(run/'public_and_actions_first2.npz') as arrays:
            obs,actions,valid=[arrays[key].copy() for key in ['observations','actions','valid_first_episode']]
        assert obs.shape==(800,2,794) and actions.shape==(800,2,18) and valid.shape==(800,2)
        assert np.isfinite(obs).all() and np.isfinite(actions).all() and abs(actions).max()<=1
        witness=torch.from_numpy(obs[valid].copy())
        portable=torch.jit.load(io.BytesIO(public['residual_actor_jit']),map_location='cpu').eval()
        with torch.inference_mode():
            expected=actor.act_inference(TensorDict({'policy':witness},batch_size=[len(witness)]))
            maximum=float((portable(witness)-expected).abs().max())
        assert maximum<=2e-5
        controller=make_rally_controller(public,2,'cpu')
        error=0.
        # The actual frozen evaluator did not reset its controller after a
        # lane ended; invalid later rows never count as another first episode.
        # A fresh first-episode controller starts at the native walk phase0.
        with torch.inference_mode():
            for observation,recorded,keep in zip(obs,actions,valid):
                action=controller.action({'policy':torch.from_numpy(observation)},default,limits)
                assert torch.isfinite(action).all() and (action.abs()<=1).all()
                if keep.any():
                    error=max(error,float(abs(action.numpy()[keep]-recorded[keep]).max()))
        assert error<=5e-4,(iteration,error)
        results.append(dict(iteration=iteration,policy_sha256=sha(policy),
            native_checkpoint_sha256=sha(checkpoint),summary=summary,
            actual_valid_public_rows=int(valid.sum()),canonical_vs_public_actor_max_error=maximum,
            CPU_vs_actual_native_action_max_error=error,all128_actual_pose_streams_finite=True,
            original_fixed_base_exact=True))
    return dict(checked_at=datetime.now(timezone.utc).isoformat(),root=str(root),
        iterations=results,bundled_RSL_used=bundled,
        scope='Four actual fixed original TRAIN128 assessments and portable weights, not formal A/B400.',
        native_replay=False,new_native_execution=False,formal_cases_read=False,
        formal_acceptance=False,goal_complete=False)


def build():
    assert not ARCHIVE.exists() and not ARCHIVE.with_suffix('.zip.partial').exists()
    assert not Path('/proc/1814619/cmdline').exists()
    assert read(ROOT/OWNER/'state.json')['state']=='COMPLETED_FRESH_BODY_RESIDUAL48_AND_FIXED_COMPARISON'
    result=check(ROOT)
    save(HERE/'ACTUAL_FOUR_DATA_CHECK.json',result)
    files,omissions={},[]

    def file(path):
        assert path.is_file() and path.resolve().is_relative_to(ROOT)
        name=path.relative_to(ROOT).as_posix()
        assert name not in files or files[name]==path
        files[name]=path

    for iteration in ITERATIONS:
        run=ROOT/PREFIX/f'body_residual_fresh_iter{iteration:02d}_fixed_train128_01'
        for path in sorted(run.rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts:
                continue
            if '.partial.' in path.name:
                final=path.with_name(path.name.replace('.partial.','.'))
                if final.exists() and sha(path)==sha(final):
                    omissions.append(dict(omitted=str(path.relative_to(ROOT)),identical_to=str(final.relative_to(ROOT)),sha256=sha(path)))
                    continue
            file(path)
        file(policy_path(ROOT,iteration))
        if iteration!=48:
            file(policy_path(ROOT,iteration).parent/'EXPORT_CHECK.json')
        for name in [f'FIXED_ITER{iteration:02d}_TRAIN128_RESULT.json',
                     f'fixed_iter{iteration:02d}_TRAIN128.csv',f'fixed_iter{iteration:02d}_TRAIN128.json']:
            file(ROOT/OWNER/name)
    for name in ['BEST_MEASURED_SOURCE.json','SOURCE_SELECTION.json','PLAN.json','state.json','export_intermediate.py']:
        file(ROOT/OWNER/name)
    for path in sorted((ROOT/COMPARE).rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:
            file(path)
    for name in ['package.py','REPORT.md','ACTUAL_FOUR_DATA_CHECK.json']:
        file(HERE/name)
    dependencies=[]
    for name,digest in LAYERS:
        path=ROOT/PREFIX/name
        assert sha(path)==digest
        dependencies.append(dict(name=name,bytes=path.stat().st_size,sha256=digest,
            url='https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/'+name))
    payloads={name:dict(bytes=path.stat().st_size,sha256=sha(path)) for name,path in files.items()}
    instructions=(
        'Four fixed policies:13,24,36,48. All original128 TRAIN cases each, original physics/targets/gates, noise none,seed724201.\n'
        'Actual complete counts:1,0,1,0. Original measured teacher remains2/128. No formal A/B400 acceptance.\n'
        'Includes all128 real50Hz body/ball pose streams per policy, real100Hz public/actions for first2 lanes with validity masks, raw episodes/contacts/plans/resets/jobs, actual public weights and source snapshots.\n'
        'Restore the two required archives in MANIFEST order into a fresh directory, then overlay this archive. The first supplies full runtime assets/URDF/walking actor; the second supplies exact source/RSL and training diagnostics/canonical48.\n'
        'For native simulation also restore runtime_external files to recorded absolute locations or equivalent links. Isaac/Kit/Torch remain external pinned environment requirements.\n'
        'Run with the restored root as the working directory so the walking actor resolves locally.\n'
        'CPU check:CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 python '+str(HERE.relative_to(ROOT))+'/package.py --check --root . --output fresh_frozen_check.json --require-bundled\n'
        'CPU replay compares actual recorded inputs/actions, not new physical trajectories. Model_i checkpoints are diagnostics; only canonical48 represents its recorded ground restart boundary.\n'
    ).encode()
    payloads['REPRODUCE.txt']=dict(bytes=len(instructions),sha256=hashlib.sha256(instructions).hexdigest())
    manifest=dict(created_at=datetime.now(timezone.utc).isoformat(),payloads=payloads,
        actual_four_checks=result,required_archives_restore_in_order=dependencies,
        duplicate_partial_omissions=omissions,exact_native_source_snapshots_retained=True,
        new_native_execution=False,formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    temporary=ARCHIVE.with_suffix('.zip.partial')
    with zipfile.ZipFile(temporary,'x',zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as output:
        for name,path in files.items():
            output.write(path,name)
        output.writestr('REPRODUCE.txt',instructions)
        output.writestr('MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(temporary) as archive:
        assert len(archive.namelist())==len(set(archive.namelist()))==len(payloads)+1
        for name,item in payloads.items():
            with archive.open(name) as stream:
                size,digest=digest_stream(stream)
            assert size==item['bytes'] and digest==item['sha256'],name
        archive.read('MANIFEST.json')
    temporary.replace(ARCHIVE)
    sealed=dict(checked_at=datetime.now(timezone.utc).isoformat(),archive=str(ARCHIVE),
        bytes=ARCHIVE.stat().st_size,sha256=sha(ARCHIVE),payloads=len(payloads),
        all_member_CRC_and_payload_SHA_verified=True,new_native_execution=False,formal_acceptance=False)
    save(HERE/'ARCHIVE_CHECK.json',sealed)
    restored=Path(tempfile.mkdtemp(prefix='g07_body48_frozen_four_restore_20261027_'))
    for path in [*[ROOT/PREFIX/name for name,_ in LAYERS],ARCHIVE]:
        with zipfile.ZipFile(path) as archive:
            assert all((restored/name).resolve().is_relative_to(restored) for name in archive.namelist())
            archive.extractall(restored)
    with (HERE/'fresh_check.log').open('x') as log:
        command=[sys.executable,str(restored/HERE.relative_to(ROOT)/'package.py'),'--check',
            '--root',str(restored),'--output',str(HERE/'FRESH_RESTORED_FOUR_DATA_CHECK.json'),'--require-bundled']
        completed=subprocess.run(command,cwd=restored,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2'),
            stdout=log,stderr=subprocess.STDOUT)
    assert completed.returncode==0
    assert sha(ARCHIVE)==sealed['sha256']
    save(HERE/'READY_FOR_UPLOAD.json',dict(checked_at=datetime.now(timezone.utc).isoformat(),
        archive=sealed,restored_root=str(restored),fresh_controller_replay_and_actor_and128_checks_passed=True,
        new_native_execution=False,formal_acceptance=False,goal_complete=False))
    print(json.dumps(sealed),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--root',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--require-bundled',action='store_true')
    args=parser.parse_args()
    if args.check:
        assert args.root and args.output
        save(args.output,check(args.root,args.require_bundled))
    else:
        build()
