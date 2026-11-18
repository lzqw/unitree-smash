#!/usr/bin/env python3
"""Verify and compact committed historical dynamics, with no optimizer or Isaac.

Each archive remains a separate stream. Original128 lane ids, unknown policy
versions, terminal observations and immutable input hashes are retained.
"""
import argparse,hashlib,io,json,subprocess,time,traceback
from pathlib import Path
import torch
from quadsmash.rl.supermimic_physical_history_v1 import compact_physical_row,historical_valid_windows


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def write(path,data):
    p=path.with_suffix(path.suffix+'.pending');p.write_text(json.dumps(data,indent=2)+'\n');p.replace(path)


def main(args):
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    reference=json.loads(args.reference_manifest.read_text());contract=reference['task_contract']
    split=json.loads(args.split_manifest.read_text());results=[];start=time.perf_counter()
    # Explicit requested paths only; never discover test fixtures as real data.
    if len(set(p.resolve() for p in args.archive))!=len(args.archive):raise ValueError('Duplicate source archive')
    for index,directory in enumerate(args.archive):
        manifest_path=directory/'manifest.json';raw=manifest_path.read_bytes();m=json.loads(raw)
        if m['schema']!='supermimic-native-physical-transition-v1' or m['num_envs']!=128:
            raise ValueError('Not an original128 native archive')
        if m['task_contract']!=contract or m['joint_names']!=reference['joint_names'] or m['control_dt_s']!=.01:
            raise ValueError('Historical physical/action/name contract mismatch')
        rows=[];meta=[];chunks=[]
        for entry in m['chunks']:
            path=directory/entry['file']
            if sha(path)!=entry['sha256']:raise ValueError('Source chunk hash mismatch: '+str(path))
            if entry.get('compression')=='zstd':
                blob=subprocess.run(['zstd','-q','-d','-c',str(path)],capture_output=True,check=True).stdout
                payload=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=True)
                del blob
            elif entry.get('compression') is None:payload=torch.load(path,map_location='cpu',weights_only=True)
            else:raise ValueError('Unknown archive compression')
            if len(payload['rows'])!=entry['ticks']:raise ValueError('Committed chunk count differs')
            for row in payload['rows']:
                values,identity=compact_physical_row(row,num_envs=128)
                rows.append(values);meta.append(identity)
            chunks.append(dict(file=entry['file'],sha256=entry['sha256'],ticks=entry['ticks']))
            del payload
            if len(chunks)%8==0:print(json.dumps(dict(archive=index,chunks=len(chunks),ticks=len(rows),optimizer_updates=0)),flush=True)
        if len(rows)!=m['control_ticks']:raise ValueError('Manifest count differs from actual rows')
        if manifest_path.read_bytes()!=raw:raise ValueError('Archive changed while reading; do not use active producer')
        data={k:torch.stack([x[k] for x in rows]) for k in rows[0]};del rows
        window=historical_valid_windows(data,[x['native_tick'] for x in meta],16)
        edges=(data['next_event_state'][...,1]>data['event_state'][...,1])|(data['next_event_state'][...,6]>data['event_state'][...,6])
        filename=f'physical_source_{index:02d}.pt';destination=args.output/filename
        torch.save(dict(schema='historical-physical-supervision-only-v1',tensors=data,native_metadata=meta,
            source_archive=str(directory.resolve()),source_manifest=m,source_manifest_sha256=hashlib.sha256(raw).hexdigest(),
            window16_valid=window,split_manifest_sha256=sha(args.split_manifest),
            eligibility=dict(world_model_supervision=True,actor_starts=False,critic_rtg=False,
                policy_execution_known=m.get('policy_execution') is not None,policy_id=m.get('policy_id'),
                reason='Recorded final commands verified; unknown policy/executor metadata never fabricated')),
            destination)
        result=dict(path=str(destination),sha256=sha(destination),source=str(directory.resolve()),source_manifest_sha256=hashlib.sha256(raw).hexdigest(),
            ticks=len(data['state']),transitions=data['state'].shape[0]*128,valid16_windows=int(window.sum()),
            contact_edges=int(edges.sum()),max_command_diff=max(x['command_max_diff'] for x in meta),
            unknown_policy_version_ticks=sum(x['policy_version'] is None for x in meta),chunks=chunks,groups={})
        for group in ('train_lanes','validation_lanes'):
            ids=torch.tensor(split[group]);result['groups'][group]=dict(lanes=len(ids),contact_edges=int(edges[:,ids].sum()),valid16_windows=int(window[:,ids].sum()))
        results.append(result);write(args.output/'progress.json',dict(status='PACKING',sources=results,optimizer_updates=0))
        print(json.dumps({k:v for k,v in result.items() if k not in ('chunks','groups')}),flush=True)
        del data,meta,window,edges
    write(args.output/'completed.json',dict(status='PACKED_VERIFIED_DYNAMICS_ONLY',sources=results,
        optimizer_updates=0,actor_updates=0,elapsed_s=time.perf_counter()-start,
        historical_fit_independence='unknown; development split does not certify past model exposure',
        split_manifest=str(args.split_manifest),split_sha256=sha(args.split_manifest)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,action='append',required=True)
    p.add_argument('--reference-manifest',type=Path,required=True);p.add_argument('--split-manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    try:main(a)
    except BaseException as e:
        if a.output.exists():write(a.output/'failure.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),optimizer_updates=0))
        raise
