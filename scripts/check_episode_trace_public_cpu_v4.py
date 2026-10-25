"""Prove lossless public-history packing on all saved native100Hz observations."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from quadsmash.rl.episode_trace_v4 import PublicHistoryPacker,unpack_public


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--run',default='residual_ppo_joint_placement_actor32_n2048_probe8_fixed_train128_01')
    args=parser.parse_args();assert not args.output.exists()
    assert Path(args.run).name==args.run, 'Require one native run name'
    run=args.source_root.resolve()/'reports/G07_full_racket_v4'/args.run
    assert json.loads((run/'job.json').read_text())['status']=='SUCCEEDED'
    path=run/'public_and_actions_first2.npz';data=np.load(path);observations=data['observations']
    assert observations.shape==(800,2,794) and observations.dtype==np.float32
    packer=PublicHistoryPacker(2);ids=np.zeros(2,dtype=np.int64)
    raw_groups=[[],[]];packed_groups=[[],[]];results=[];detected_resets=0
    def close(lane):
        if not raw_groups[lane]:return
        raw=np.stack(raw_groups[lane]);compact=np.stack(packed_groups[lane]);expanded=unpack_public(compact)
        assert np.array_equal(raw,expanded)
        results.append(dict(lane=lane,inferred_local_episode_index=int(ids[lane]),rows=len(raw),
            full_public_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
            reconstructed_sha256=hashlib.sha256(expanded.tobytes()).hexdigest(),bitwise_equal=True))
        raw_groups[lane]=[];packed_groups[lane]=[]
    for tick,obs in enumerate(observations):
        if tick:
            for lane in np.flatnonzero(obs[:,775]<observations[tick-1,:,775]-1e-6):
                close(int(lane));ids[lane]+=1;detected_resets+=1
        compact,_=packer.pack(obs,ids)
        for lane in range(2):raw_groups[lane].append(obs[lane].copy());packed_groups[lane].append(compact[lane].copy())
    for lane in range(2):close(lane)
    assert sum(row['rows']for row in results)==1600
    count,steps,bodies=2048,801,28
    required=count*steps*(104+54+bodies*7+3)*4+count*steps*4+count*10*69*4
    result=dict(checked_at=datetime.now(timezone.utc).isoformat(),
        scope='Lossless CPU encoding/reconstruction of actual saved native public inputs. Not a new policy execution or physical trajectory generated from these inputs.',
        actual_observation_rows=1600,source_observation_shape=list(observations.shape),
        stored_per_control_values=104,reconstructed_public_values=794,all_full_public_values_bitwise_equal=True,
        per_inferred_episode=results,actual_clock_resets_observed=detected_resets,
        automatic_reset_checked_on_real_rows=bool(detected_resets),
        nominal2048_lane801_step28_body_ring_buffer_bytes=required,
        nominal_buffer_GiB=required/2**30,
        recorder_selection='First24 complete episodes, first6 failures and first episodes of lanes0/8/19/23/58/107; all raw episode outcomes remain in normal native logs.',
        native_recorder_integrated_in_candidate=True,new_native_recording_executed=False,
        limitations=['Only public input packing was checked on real native observations here. Body/ball/action capture integration still needs a real native candidate run.',
            ('No real clock reset occurs in this source; controlled fixtures cover selective resets.' if detected_resets==0 else
             'Actual saved clock resets are included in this lossless public-input check. Complete live body/ball/action producer hooks have still not run in a native candidate job.'),
            'Memory estimate covers main NumPy buffers and history, excluding transient copies and mesh-export data.'],
        source_sha256={str(path):sha(path),str(ROOT/'quadsmash/rl/episode_trace_v4.py'):sha(ROOT/'quadsmash/rl/episode_trace_v4.py'),
            str(Path(__file__).resolve()):sha(Path(__file__).resolve())},
        native_execution=False,formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix('.json.tmp');temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(args.output)
    print(json.dumps(dict(actual_rows=1600,bitwise_public_reconstruction=True,episodes_checked=len(results),
        nominal_ring_buffer_GiB=result['nominal_buffer_GiB'],new_native_recording_executed=False)),flush=True)


if __name__=='__main__':main()
