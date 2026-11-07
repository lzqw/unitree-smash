#!/usr/bin/env python3
"""Compare executed learner trajectories with verified demonstrations.

Time-aligned teacher commands after state divergence are not counterfactual
expert labels. This audit diagnoses prefix preservation and distribution shift;
it never adds failed student actions to a positive demonstration dataset.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_native_recovery_command_v1 import load_episodes
from quadsmash.control.learned_recovery_command_v1 import command_features
from quadsmash.rl.episode_trace_v4 import unpack_public


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def first_above(values,threshold):
    positions=np.flatnonzero(values>threshold)
    return int(positions[0]) if len(positions) else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--command-audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise ValueError('Preserve prior evidence')
    torch.set_num_threads(1)
    audit=json.loads(args.command_audit.read_text())
    if not audit['passed'] or Path(audit['run']).resolve()!=args.run.resolve():
        raise ValueError('Require completed matching native execution audit')
    for name,digest in audit['source_sha256'].items():
        if sha(name)!=digest:raise ValueError('Changed native evidence: '+name)
    inputs=json.loads((args.run/'INPUTS.json').read_text())
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if (sha(args.checkpoint)!=inputs['learned_recovery_command_head_sha256'] or
        checkpoint['dataset_manifest_sha256']!=sha(args.dataset/'manifest.json')):
        raise ValueError('Native head or verified training dataset differs')
    manifest,episodes,limits=load_episodes(args.dataset)
    scales=checkpoint['head_state_dict']['feature_scale'].numpy()
    means=checkpoint['head_state_dict']['feature_mean'].numpy()
    widths=(limits[:,1]-limits[:,0]).numpy()
    traces=args.run/'fixed_policy_traces'
    native_entries={r['source_training_index']:r for r in json.loads((traces/'manifest.json').read_text())['episodes']}
    data_entries={r['case']:r for r in manifest['episodes']}
    outcome={r['source_training_index']:r['contact_flight_recovery']
             for r in json.loads((args.run/'episodes.json').read_text())}
    with np.load(args.run/'learned_recovery_commands.npz',allow_pickle=False) as z:
        history={k:z[k].copy() for k in ('base_command','final_command','active','first_episode')}
    rows=[];hashes={}
    for e in episodes:
        case=e['case'];entry=native_entries[case];path=traces/entry['data_file']
        if sha(path)!=entry['data_sha256']:raise ValueError('Changed native trajectory')
        hashes[str(path.resolve())]=sha(path)
        with np.load(path,allow_pickle=False) as z:
            public=unpack_public(z['packed_public']);actual=z['dispatched_actions'].copy()
        source=args.dataset/data_entries[case]['data_file']
        with np.load(source,allow_pickle=False) as z:
            teacher_public=unpack_public(z['packed_public'])
            teacher_actions=z['native_action'].copy()
            teacher_ticks=np.flatnonzero(z['supervision_mask'])
        hashes[str(source.resolve())]=sha(source)
        active=np.flatnonzero(history['active'][:len(public),case]&history['first_episode'][:len(public),case])
        if not len(active):
            rows.append(dict(case=case,corrected=e['corrected'],active=False));continue
        onset=int(active[0]);teacher_onset=int(teacher_ticks[0]);n=min(len(public),len(teacher_public))
        if onset==0:raise ValueError('No previous command at onset')
        previous=np.vstack((actual[:1],actual[:-1]))
        features=command_features(torch.from_numpy(public),
            torch.from_numpy(history['base_command'][:len(public),case]),torch.from_numpy(previous),
            torch.from_numpy(public[:,775]-public[onset,775])).numpy()
        public_difference=np.max(abs(public[:n]-teacher_public[:n]),axis=1)
        command_difference=np.max(abs(actual[:n]-teacher_actions[:n])*widths/2.,axis=1)
        # Compare same control ticks, not merely the nth active frame.
        joined=[(int(t),i) for i,t in enumerate(teacher_ticks) if t<len(public) and history['active'][t,case]]
        fs=np.asarray([t for t,_ in joined],dtype=int);ts=np.asarray([i for _,i in joined],dtype=int)
        if not len(joined):raise ValueError('No corresponding recovery timestamps')
        feature_error=abs(features[fs]-e['features'].numpy()[ts])/scales
        early=(fs>=onset)&(fs<onset+100)
        first_matching=np.flatnonzero(fs==onset)
        onset_feature_error=(float(feature_error[first_matching[0]].max()) if len(first_matching) else None)
        physical=outcome[case]
        rows.append(dict(case=case,corrected=e['corrected'],active=True,
            teacher_outcome=e['outcome'],native_onset_tick=onset,teacher_onset_tick=teacher_onset,
            pre_onset_public_max_abs=float(public_difference[:min(onset,teacher_onset,n)].max(initial=0.)),
            pre_onset_command_max_rad=float(command_difference[:min(onset,teacher_onset,n)].max(initial=0.)),
            first_public_difference_tick_above_1e_5=first_above(public_difference,1e-5),
            first_command_difference_tick_above_1e_5_rad=first_above(command_difference,1e-5),
            onset_time_aligned_command_difference_max_rad=float(command_difference[onset]) if onset<n else None,
            onset_feature_difference_max_training_scale=onset_feature_error,
            first_second_feature_difference_max_training_scale=float(feature_error[early].max()) if early.any() else None,
            first_second_feature_difference_p95_training_scale=float(np.quantile(feature_error[early],.95)) if early.any() else None,
            active_feature_clip_fraction=float(np.mean(abs((features[active]-means)/scales)>8.)),
            body_landing=physical['body_landing'],stable_recovery_s=physical['recovery_s'],
            terminal_root_height_m=physical['terminal_root_height_m']))
    for f in [Path(__file__),args.command_audit,args.checkpoint,args.dataset/'manifest.json',
              args.run/'INPUTS.json',args.run/'episodes.json',args.run/'learned_recovery_commands.npz',
              ROOT/'scripts/train_native_recovery_command_v1.py']:
        hashes[str(f.resolve())]=sha(f)
    result=dict(schema='native-recovery-learning-state-drift-v1',run=str(args.run.resolve()),
        verified_dataset_episodes=len(rows),cases=rows,source_sha256=hashes,
        scope=__doc__,thresholds=dict(public_abs=1e-5,command_rad=1e-5),
        feature_scale_is_training_normalizer=True,normalized_distance_is_failure_risk=False,
        positive_labels_generated=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(episodes=len(rows),active=sum(r['active'] for r in rows),output=str(args.output))))


if __name__=='__main__':main()
