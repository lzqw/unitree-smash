#!/usr/bin/env python3
"""Check dataset eligibility, total-command/history mapping and expressibility.

No policy training or teacher-off evaluation. Existing legacy residual bounds
are not relabelled: the checked native-delta coordinate is explicitly new.
"""
import argparse,hashlib,json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from quadsmash.rl.episode_trace_v4 import unpack_public
from quadsmash.rl.recovery_command_dataset_v1 import recovery_eligibility,native_commands,compose_native_recovery_delta


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve previous audits')
    manifest_path=a.dataset/'manifest.json';manifest=json.loads(manifest_path.read_text())
    if manifest['schema']!='verified-recovery-native-command-dataset-v1':raise ValueError('Unsupported schema')
    limits=np.asarray(manifest['joint_limits_rad']);width=limits[:,1]-limits[:,0]
    rows=[]
    for entry in manifest['episodes']:
        path=a.dataset/entry['data_file']
        if sha(path)!=entry['data_sha256']:raise ValueError('Changed positive command file')
        with np.load(path,allow_pickle=False) as data:
            public=unpack_public(data['packed_public'])
            if hashlib.sha256(public.tobytes()).hexdigest()!=entry['full_public794_sha256']:
                raise ValueError('Changed actual observation/history')
            active,elapsed=recovery_eligibility(public)
            np.testing.assert_array_equal(active,data['recovery_eligible'])
            np.testing.assert_allclose(elapsed,data['elapsed_since_recovery_s'],atol=1e-8)
            mask=data['supervision_mask'];known=data['previous_command_known']
            np.testing.assert_array_equal(mask,active&known)
            target=data['native_action'][mask].astype(float)
            previous_q=data['previous_joint_target_rad'][mask]
            previous=2.*(previous_q-limits[:,0])/width-1.
            _,all_q=native_commands(data['native_action'],limits)
            np.testing.assert_allclose(data['previous_joint_target_rad'][1:],all_q[:-1],atol=1e-6)
            np.testing.assert_allclose(data['native_joint_target_rad'],all_q,atol=1e-6)
            roles=data['label_role'][mask];known_request=data['student_request_known'][mask]
            if np.any((roles==2)&~data['teacher_applied'][mask]):raise ValueError('False teacher label')
            if np.any((roles==3)&data['teacher_applied'][mask]):raise ValueError('False fallback label')
            if np.any((roles!=1)&~known_request):raise ValueError('Missing recorded student request')
            # For unmodified rows only, the executed command is the baseline
            # command. It is never labelled as the raw network residual.
            base=target.copy()
            base[known_request]=2.*(data['student_requested_joint_target_rad'][mask][known_request]-limits[:,0])/width-1.
            delta=(target-base)/2.
            if np.max(abs(delta),initial=0.)>1.+1e-6:raise ValueError('Target outside declared new action coordinate')
            reconstructed=compose_native_recovery_delta(base,delta,previous,np.ones(len(target),dtype=bool),
                limits,manifest['arm_target_rate_rad_s'],manifest['control_dt_s'])
            error=float(np.max(abs(reconstructed-target)*width/2.,initial=0.))
            if error>5e-6:raise ValueError('Native total command cannot be reproduced by proposed coordinate')
            rows.append(dict(case=entry['case'],outcome=entry['outcome'],rows=int(mask.sum()),
                correction_rows=int((roles==2).sum()),fallback_rows=int((roles==3).sum()),
                native_target_roundtrip_error_max_rad=error,delta_abs_max=float(np.max(abs(delta),initial=0.))))
    result=dict(schema='recovery-command-coordinate-audit-v1',scope=__doc__,dataset=str(a.dataset.resolve()),
        passed=True,episodes=len(rows),supervision_rows=sum(r['rows'] for r in rows),
        correction_rows=sum(r['correction_rows'] for r in rows),
        maximum_native_target_roundtrip_error_rad=max(r['native_target_roundtrip_error_max_rad'] for r in rows),
        samples_include_case_id_as_policy_input=False,future_outcomes_used_as_policy_input=False,
        policy_training_performed=False,teacher_off_execution_performed=False,
        episodes_detail=rows,source_sha256={str(p.resolve()):sha(p) for p in [manifest_path,Path(__file__),
            ROOT/'quadsmash/rl/recovery_command_dataset_v1.py']})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','episodes_detail','scope']}))


if __name__=='__main__':main()
