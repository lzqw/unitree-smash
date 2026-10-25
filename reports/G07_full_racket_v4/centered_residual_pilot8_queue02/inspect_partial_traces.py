"""Inspect saved traces from the still-running native job without certifying it.

The final audit additionally requires a successful job and exact equality to
all original terminal/contact records. This observation cannot replace it.
"""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from quadsmash.rl.episode_trace_v4 import unpack_public
from quadsmash.evaluation_v4 import independent_episode


def main():
    run=ROOT/'reports/G07_full_racket_v4/residual_ppo_centered_base8_n2048_probe8_02'
    output=HERE/'PARTIAL_NATIVE_TRACE_CAPTURE_CHECK.json';assert not output.exists()
    assert Path('/proc/1677688').exists()and not(run/'job.json').exists()
    directory=run/'trajectory_traces';snapshot=(directory/'manifest.json').read_bytes();manifest=json.loads(snapshot)
    assert manifest['enabled']and manifest['disabled_error']is None and manifest['native_execution']
    assert manifest['provenance']['source_job_output']==str(run)
    protocol=json.loads((ROOT/'configs/g07_full_racket_eval_v4_r3.json').read_text());protocol['_root']=str(ROOT)
    names=manifest['body_names'];tool=names.index('Racket');checked=[];sources={}
    for entry in manifest['saved_episodes']:
        path=directory/entry['data_file'];metadata_path=path.with_suffix('.json');metadata=json.loads(metadata_path.read_text())
        assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['data_sha256']==metadata['data_sha256']
        with np.load(path)as stream:
            data={k:stream[k]for k in stream.files}
        public=unpack_public(data['packed_public']);length=len(public);row=metadata['raw_episode']
        assert length==entry['samples']==metadata['samples']
        assert hashlib.sha256(public.tobytes()).hexdigest()==entry['full_public794_sha256']==metadata['full_public794_sha256']
        assert abs(public[0,775])<=1e-6 and np.max(abs(np.diff(public[:,775])-.01))<=2e-6
        assert abs(float(public[-1,775])+.01-row['terminal_native_state']['time_s'])<=2e-5
        poses=data['link_poses'];assert poses.shape==(length,len(names),7)and np.isfinite(poses).all()
        inverse=Rotation.from_quat(poses[:,0,3:][:,[1,2,3,0]]).inv()
        actual_tool=inverse.apply(poses[:,tool,:3]-poses[:,0,:3])
        tool_error=float(np.linalg.norm(actual_tool-public[:,45:48],axis=1).max())
        rotation=Rotation.from_quat(poses[:,tool,3:][:,[1,2,3,0]])
        normal=inverse.apply(rotation.apply(np.tile([1.,0.,0.],(length,1))))
        normal_error=float(np.linalg.norm(normal-public[:,48:51],axis=1).max())
        assert tool_error<=2e-4 and normal_error<=5e-5
        assert all(data[k].shape==(length,18)and np.isfinite(data[k]).all()for k in ['residual_actions','base_actions','native_actions'])
        assert np.max(abs(data['native_actions']))<=1.000001
        assert np.unique(data['learner_iteration']).tolist()==entry['actor_iterations_encountered']
        assert row['ground_init']and row['source_training_index']==entry['source_training_index']
        outcome=independent_episode(row,protocol);assert outcome['complete']==bool(row['complete_smash'])
        checked.append(dict(env_id=entry['env_id'],episode_index=entry['episode_index'],samples=length,
            actor_iterations_encountered=entry['actor_iterations_encountered'],metadata_outcome=outcome,
            public_pose_error_m=tool_error,public_normal_error=normal_error,full_public_SHA_verified=True))
        for source in [path,metadata_path]:sources[str(source.relative_to(ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
    assert checked
    sources[str(Path(__file__).relative_to(ROOT))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    value=dict(checked_at=datetime.now(timezone.utc).isoformat(),
        scope='Partial observation of saved actual100Hz ground-start training episodes while native GPU0 job is still live. Not a final trace audit, job completion, frozen score or acceptance.',
        native_process_verified_live=True,actual_saved_traces_checked=len(checked),episodes=checked,
        observed_manifest_sha256=hashlib.sha256(snapshot).hexdigest(),observed_manifest_snapshot=manifest,
        source_sha256=sources,source_job_and_full_terminal_contact_record_comparison_pending=True,
        diagnostic_tolerances_are_not_acceptance_thresholds=True,formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    tmp=output.with_suffix('.json.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(output)
    print(json.dumps(dict(actual_saved_traces_checked=len(checked),full_public_sha_verified=True,source_job_still_live=True,final_audit_pending=True)))


if __name__=='__main__':main()
