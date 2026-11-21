#!/usr/bin/env python3
"""Replay a first Recovery block's real WM batches with a prescribed learning rate."""
import argparse
import json
from pathlib import Path

import torch

from scripts.evaluate_supermimic_fresh_wm_panel import digest
from quadsmash.rl.supermimic_archive_storage_v1 import read_archive_rows
from quadsmash.rl.supermimic_physical_imagination_state_v1 import PhysicalImaginationStateV1
from quadsmash.rl.supermimic_physical_learner_v1 import PhysicalHorizonTrainer
from quadsmash.rl.supermimic_physical_loop_loading_v1 import load_physical_loop, tensor_digest
from quadsmash.rl.supermimic_physical_loop_replay_v1 import NaturalFlightWindows, PhysicalLoopReplay
from quadsmash.rl.supermimic_physical_start_cache_v1 import load_start_cache, APPROACH_TICKS
from quadsmash.rl.supermimic_timebase_v2 import Timebase
from quadsmash.rl.supermimic_training_probe_replay import import_training_collection


def read(path):
    return json.loads(path.read_text())


def fit_readout_rows(module, layer, rows):
    """Use the calibrated encoder and allow Adam to fit only the named output rows."""
    module.requires_grad_(False)
    layer.requires_grad_(True)
    for parameter in layer.parameters():
        mask = torch.zeros_like(parameter)
        mask[rows] = 1
        parameter.register_hook(lambda grad, active=mask: grad*active)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run','checkpoint','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--learning-rate',type=float,required=True)
    parser.add_argument('--separate-supervision', action='store_true')
    parser.add_argument('--natural-flight-batch',type=int,default=0,help='Auxiliary 64-step native training windows per WM update')
    parser.add_argument('--adaptation',choices=('full','joint-and-racket-readouts'),default='full')
    parser.add_argument('--device',choices=('cpu','cuda:0'),default='cpu')
    parser.add_argument('--cuda-memory-gib',type=float,help='Limit allocator usage when sharing the GPU with a native rollout')
    parser.add_argument('--cpu-threads',type=int,default=5)
    args = parser.parse_args()
    if args.natural_flight_batch<0:
        parser.error('Natural flight batch must be nonnegative')
    if not 0 < args.learning_rate < 1 or args.cpu_threads <= 0:
        parser.error('Expected positive CPU count and learning rate in (0,1)')
    torch.set_num_threads(args.cpu_threads)
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable')
    if args.cuda_memory_gib is not None:
        if device.type!='cuda' or not 0 < args.cuda_memory_gib < float('inf'):
            parser.error('A positive CUDA allocator limit requires CUDA')
        total=torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(args.cuda_memory_gib*(1<<30)/total,1.),device)
    cfg = read(args.run/'config.json')
    argv = read(args.run/'startup.json')['argv']
    ck = torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    stages = []
    for line in (args.run/'training_history.jsonl').read_text().splitlines():
        row = json.loads(line)
        stages.append(row)
        if row['world_update_reports'][-1]['world_updates'] == ck['world_updates']:
            break
    if (not stages or stages[-1]['world_update_reports'][-1]['world_updates'] != ck['world_updates']
            or any(s['block_name']!='recovery' or s['attempt_id']!=i+1 for i,s in enumerate(stages))
            or '--resume-training' in argv or '--wm-bootstrap-max-ticks' in argv):
        raise ValueError('Expected an original uninterrupted first Recovery block without bootstrap')
    def option(key):
        return Path(argv[argv.index('--'+key)+1])
    def number(key, fallback):
        return float(argv[argv.index('--'+key)+1]) if '--'+key in argv else fallback
    def identity(data):
        return tensor_digest(data.items())
    wm=ck['metadata']['physical_imagination']
    paths=wm['checkpoint_paths']
    archive=option('archive')
    actor,adapter,template,launch,manifest=load_physical_loop(archive=archive,
        physical_checkpoint=paths['physical'],ball_checkpoint=paths['ball'],task_checkpoint=paths['task'],
        action_ball_checkpoint=paths.get('action_ball'),safety_mapping=option('safety-mapping'),
        execution_root=wm['g07_execution_root'],device=device)
    if wm['checkpoint_sha256'] != adapter.manifest['checkpoint_sha256']:
        raise ValueError('Initial WM dependencies differ from the live stage')
    packed_path=option('packed-data')
    packed_sha=digest(packed_path)
    if packed_sha != read(packed_path.parent/'completed.json')['sha256']:
        raise ValueError('Packed training archive changed')
    packed=torch.load(packed_path,map_location='cpu',weights_only=True,mmap=True)
    if packed['source_manifest']['task_contract']!=manifest['task_contract']:
        raise ValueError('Training data task mismatch')
    n_ticks=stages[-1]['native_ticks']
    replay=PhysicalLoopReplay(packed,adapter.task_model.memory_schema,
        capacity_ticks=len(packed['tensors']['state'])+n_ticks+1)
    imports=[]
    held=cfg['wm_holdout_lanes']
    natural=NaturalFlightWindows(packed['tensors'],held) if args.natural_flight_batch else None
    for i,flag in enumerate(argv):
        if flag=='--wm-action-collection':
            imports.append(import_training_collection(replay,Path(argv[i+1]),manifest['task_contract'],manifest['policy_execution'],held))
    replay.new_stream(source='online',policy_version=0)
    timebase=Timebase(**read(Path(__file__).resolve().parents[1]/'configs/supermimic_task_horizon_v2.json'))
    starts=load_start_cache(archive,manifest)
    state=PhysicalImaginationStateV1.from_record(starts[APPROACH_TICKS[0]],executor_template=template,launch=launch,device=device)
    trainer=PhysicalHorizonTrainer(actor,adapter,state,timebase,world_lr=args.learning_rate)
    trainer.ball_loss_weight=number('wm-ball-loss-weight',4.)
    trainer.contact_ball_loss_weight=number('wm-contact-ball-loss-weight',8.)
    trainer.racket_loss_weight=number('racket-loss-weight',1.)
    initial=identity(adapter.state_dict())
    if args.adaptation=='joint-and-racket-readouts':
        from quadsmash.rl.supermimic_ball_residual_v2 import RacketSweepSupportedBallV2
        from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2
        body, ball = adapter.physical_model, adapter.ball_model
        if (not isinstance(body,PhysicalWorldV2) or not isinstance(ball,RacketSweepSupportedBallV2)
                or adapter.action_ball_model is not None):
            raise ValueError('Expected calibrated 55D body and object-separated sweep ball')
        fit_readout_rows(body,body.body.net[-1],slice(13,49))
        fit_readout_rows(ball,ball.response.net[-1],slice(0,6))
    protected = {k:v.detach().clone() for k,v in adapter.named_parameters()
                 if args.adaptation!='full' and not v.requires_grad}
    protected_rows = {}
    if args.adaptation!='full':
        for component,layer,rows_to_keep in ((body,body.body.net[-1],list(range(13))+list(range(49,59))),
                                              (ball,ball.response.net[-1],list(range(6,24)))):
            for param in layer.parameters():
                protected_rows[param]=(rows_to_keep,param[rows_to_keep].detach().clone())
    native=args.run/'native_transitions'
    observed=read(native/'manifest.json')
    if observed['task_contract']!=manifest['task_contract'] or observed['policy_execution']!=manifest['policy_execution']:
        raise ValueError('Online data task or executor changed')
    args.output.mkdir(parents=True,exist_ok=False)
    hashes=[]
    def native_rows():
        for item in observed['chunks']:
            with read_archive_rows(native,item,scratch=args.output) as rows:
                hashes.append(item['sha256'])
                yield from rows
    rows=native_rows()
    tick=0
    try:
        with (args.output/'progress.jsonl').open('x') as f:
            for stage in stages:
                while tick<stage['native_ticks']:
                    row=next(rows)
                    if (row['native_tick']!=tick or row['collection_mode']!='mixed_training'
                            or row['source_id']!='online' or row['stream_id']!=replay.current_stream):
                        raise ValueError(f'Different native source or stream at tick {tick}')
                    replay.append(row,retain_start=False,policy_version=row['policy_version'],native_tick=tick)
                    tick+=1
                reports=stage['world_update_reports']
                for offset, observed_step in enumerate(reports):
                    window,lanes,hist=replay.window(timebase.world_steps,cfg['world_batch'],
                        stage['attempt_id']*len(reports)+offset,device,source_quota=cfg['wm_source_quota'],
                        stage_quota=cfg['wm_stage_quota'],head='recovery',excluded_lanes=held,
                        recent_policy_window=cfg['wm_recent_policy_window'],recent_tick_window=cfg['wm_recent_tick_window'])
                    for key in ('source_counts','stage_counts','stage_actual_counts','actual_event_coverage','fallback_count',
                                'unique_episode_count','native_tick_min','native_tick_max'):
                        if hist[key]!=observed_step['sampling'][key]:
                            raise ValueError(f'Sampled {key} differs at WM update {trainer.world_updates+1}')
                    long_native=None
                    long_sha=None
                    if natural is not None:
                        extra,extra_lanes=natural.sample(args.natural_flight_batch,trainer.world_updates,device)
                        long_native=(extra,launch[extra_lanes])
                        long_sha=identity(extra)
                    metrics=trainer.fit_world_window(window,launch[lanes],separate_supervision=args.separate_supervision,long_native=long_native)
                    loss=metrics['world_loss']
                    if trainer.world_updates==1 and long_native is None and not args.separate_supervision and abs(loss-observed_step['world_loss'])>0.001*max(1,abs(loss)):
                        raise ValueError(f'First objective does not reproduce native update: {loss} vs {observed_step["world_loss"]}')
                    info=dict(update=trainer.world_updates,loss=loss,original_loss=observed_step['world_loss'],
                              window_sha256=identity(window),natural_window_sha256=long_sha,flight_loss=metrics['long_native_flight_loss'],fallback=hist['fallback_count'])
                    f.write(json.dumps(info)+'\n')
                    f.flush()
    finally:
        rows.close()
    for name, param in adapter.named_parameters():
        if name in protected and not torch.equal(param,protected[name]):
            raise RuntimeError(f'Unexpected update to frozen parameter {name}')
    for parameter,(rows_to_keep,before) in protected_rows.items():
        if not torch.equal(before,parameter[rows_to_keep]):
            raise RuntimeError('Unexpected update in a protected output row')
    revised=adapter.state_dict()
    identity_after=identity(revised)
    if identity_after==initial or trainer.world_updates!=ck['world_updates']:
        raise RuntimeError('No complete online update was produced')
    out=args.output/'online_world.pt'
    torch.save(revised,out.with_suffix('.pending'))
    out.with_suffix('.pending').replace(out)
    info=dict(device=str(device),peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
              updates=trainer.world_updates,policy_updates=0,critic_updates=0,natural_flight_batch=args.natural_flight_batch,
              original_learning_rate=cfg['world_lr'],learning_rate=args.learning_rate,adaptation=args.adaptation,separate_supervision=args.separate_supervision,
              stage_checkpoint_sha256=digest(args.checkpoint),original_online_world_sha256=identity(ck['world']),
              initial_world_sha256=initial,online_world_sha256=identity_after,
              chunks=hashes,packed_sha256=packed_sha,training_collection_receipts=imports,excluded_lanes=held,
              scope='Offline replay from the original component weights; development candidate only.')
    (args.output/'fit.json').write_text(json.dumps(info,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
