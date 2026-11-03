#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path
import numpy as np

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--historical-targets',type=Path,required=True)
    p.add_argument('--candidate-retention',type=Path,required=True)
    p.add_argument('--candidate-repeat',type=int,default=4)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise ValueError('fresh output required')
    if a.candidate_repeat<1: raise ValueError('positive candidate repeat')
    hm=json.loads((a.historical_targets/'manifest.json').read_text())
    cm=json.loads((a.candidate_retention/'manifest.json').read_text())
    arrays=[]
    episode_sources=[]
    episode_labels=[]
    sources=[]
    for kind,root,m,repeat in [('historical',a.historical_targets,hm,1),('candidate',a.candidate_retention,cm,a.candidate_repeat)]:
        with np.load(root/m['data_file'],allow_pickle=False) as z:
            x=z['observations'].copy(); y=z['residual_targets'].copy(); off=z['episode_offsets'].copy()
        if x.shape[1:]!=(794,) or y.shape!=(len(x),18): raise ValueError('invalid source arrays')
        source_index=len(sources)
        sources.extend(m['sources'])
        for rep in range(repeat):
            for i in range(len(off)-1):
                st,en=int(off[i]),int(off[i+1])
                arrays.append((x[st:en],y[st:en]))
                episode_sources.append(source_index)
                episode_labels.append(dict(source_kind=kind,source_episode=int(i),repeat=int(rep),
                                          source_training_index=m['episodes'][i].get('source_training_index')))
    observations=np.concatenate([x for x,_ in arrays],axis=0)
    targets=np.concatenate([y for _,y in arrays],axis=0)
    offsets=[0]
    for x,_ in arrays: offsets.append(offsets[-1]+len(x))
    a.output.mkdir(parents=True)
    data=a.output/'residual_retention.npz'
    np.savez_compressed(data,observations=observations,residual_targets=targets,
                        episode_offsets=np.asarray(offsets,dtype=np.int64),
                        episode_source_index=np.asarray(episode_sources,dtype=np.int64))
    episodes=[dict(label, samples=int(offsets[i+1]-offsets[i]), command_replay_validated=True)
              for i,label in enumerate(episode_labels)]
    manifest=dict(schema='verified-recovery-repair-residual-retention-v1',
        sources=sources,episodes=episodes,episode_count=len(episodes),samples=len(observations),
        minimum_speed_mps=5.,protocol_sha256=hm['protocol_sha256'],
        target_interface='raw-residual18-before-mixer',data_file=data.name,
        data_sha256=sha(data),label_validation='recovery-head-targets-v1',
        cpu_gpu_command_tolerance=5e-4,candidate_policy_sha256=cm.get('candidate_policy_sha256'),
        historical_manifest_sha256=sha(a.historical_targets/'manifest.json'),
        candidate_manifest_sha256=sha(a.candidate_retention/'manifest.json'),
        candidate_repeat=a.candidate_repeat,phase_gate='raw_public_phase774_gt_0.5',
        action_override_at_deployment=False,formal_acceptance=False,
        online_selector_evaluated=False,
        scope='Mixture of historical phase774 repair targets and exact current-candidate successful targets. '
              'Candidate episodes are repeated for retention weighting; this remains supervised guidance, '
              'not an online multi-policy selector or a new physical success result.')
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(episodes=len(episodes),historical_episodes=len(hm['episodes']),
                          candidate_episodes=len(cm['episodes'])*a.candidate_repeat,
                          samples=len(observations),manifest_sha256=sha(a.output/'manifest.json'))))
if __name__=='__main__': main()
