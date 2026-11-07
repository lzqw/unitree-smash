"""Prepare an original-ground early-feedback run with passive head/load evidence."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def prepare(root,output):
    root,output=Path(root).resolve(),Path(output).resolve()
    if output.exists():raise ValueError('Fresh preparation required')
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    source=root/'reports/landing_episode_native_followup_20261107/entry/evaluate_sequence.py'
    audit_source=root/'scripts/audit_native_recovery_sequence_v2.py'
    if sha(source)!='37c6d3077b74388b6fa0c5cfa6b4f571ea5eff823bb8713638da5e987d363ad5':raise ValueError('Changed native template')
    if sha(audit_source)!='01667d30c5ba5cbcb323c5902bd56bebcf0fc8e54f917500f5140f56edb20953':raise ValueError('Changed audit template')
    text=source.read_text()
    attach='        finished = torch.zeros(count, dtype=torch.bool, device=env.device)'
    insert=('        from scripts.native_isolation_microtrace_v1 import NativeIsolationMicrotraceV1\n'
        '        isolation_observer=NativeIsolationMicrotraceV1(env,args.output,lanes=(3,125),first_frame=89600,last_frame=92000)\n'
        '        from scripts.native_head_ground_observation_v1 import NativeHeadGroundObservationV1\n'
        '        head_observer=NativeHeadGroundObservationV1(env,args.output)\n')
    replacements=[
        ('from scripts.native_recovery_sequence_replay_v2 import NativeRecoverySequenceReplayV2 as NativeRecoverySequenceReplay',
         'from scripts.native_early_support_tail_v1 import NativeEarlySupportTailV1 as NativeRecoverySequenceReplay'),
        (attach,insert+attach),
        ('    finally:\n        env._diagnostic_observer=None',
         "    finally:\n        if 'head_observer' in locals():\n            head_observer.save()\n            head_observer.detach()\n"
         "        if 'isolation_observer' in locals():\n            isolation_observer.save()\n            isolation_observer.detach()\n"
         '        env._diagnostic_observer=None'),
        ('def main():\n    args.output.mkdir(parents=True, exist_ok=True)',
         'storage_reserve=None\n\ndef main():\n    global storage_reserve\n    args.output.mkdir(parents=True, exist_ok=True)\n'
         '    from scripts.native_output_reserve_v1 import NativeOutputReserve\n    storage_reserve=NativeOutputReserve(args.output)'),
        ('            for tick in range(requested):','            for tick in range(requested):\n                storage_reserve.check()'),
        ('        contact_observer.save()','        storage_reserve.release()\n        contact_observer.save()'),
        ('finally:\n    app.close(skip_cleanup=True)',
         'finally:\n    if storage_reserve is not None:\n        storage_reserve.release()\n    app.close(skip_cleanup=True)'),
        ('deployable_risk_trigger=False,sustained_mpc_feedback_implemented=False,',
         'deployable_risk_trigger=False,sustained_mpc_feedback_implemented=False,\n'
         '            measured_state_feedback=True,offline_optimized_reference=True,')]
    for before,after in replacements:
        if text.count(before)!=1:raise ValueError('Changed integration contract '+before)
        text=text.replace(before,after)
    audit=audit_source.read_text()
    replacements=[
        ('ROOT=Path(__file__).resolve().parents[1]',f'ROOT=Path({str(root)!r})'),
        ("    if (result['applied_steps']!=steps or inputs['landing_mpc_cases'] is not None or",
         "    if result.get('early_termination_tick') is not None:\n"
         "        rejection=dict(passed_dispatch_integrity=False,native_execution=True,complete5=metrics['complete5'],\n"
         "            reason='Selected original episode ended before the planned feedback horizon; negative result retained.',\n"
         "            early_termination_tick=result['early_termination_tick'],applied_steps=result['applied_steps'],\n"
         "            planned_steps=steps,valid_expert_labels=False,teacher_off_learning_result=False,\n"
         "            source_sha256={str(a.run/name):sha(a.run/name) for name in ['job.json','INPUTS.json','metrics.json','sequence_native_result.json']})\n"
         "        a.output.write_text(json.dumps(rejection,indent=2)+'\\n')\n"
         "        raise SystemExit(2)\n"
         "    if (result['applied_steps']!=steps or inputs['landing_mpc_cases'] is not None or"),
        ("    command_error=float(np.max(abs(targets[start:start+steps]-planned)))\n    if command_error>1e-6:raise ValueError('Planned sequence changed before native dispatch')",
         '    from scripts.check_early_support_dispatch_v1 import verify_feedback_dispatch\n'
         '    feedback_check=verify_feedback_dispatch(a.run,a.bundle)\n'
         "    command_error=feedback_check['maximum_target_error_rad']"),
        ('common=min(start,n,len(ba))','common=min(start,n,len(ba)) if lane==case else min(n,len(ba))\n'
         "            if lane!=case and n!=len(ba):raise ValueError('Untouched first episode length changed')"),
        ('planned_command_mapping_error_rad=command_error,','planned_command_mapping_error_rad=command_error,\n'
         '        measured_state_feedback_dispatch=feedback_check,online_mpc=False,\n'
         "        command_reference='Feedback recomputed from actual state; nominal sequence is not an executed-action label',"),
        ("files += [Path(__file__),archive,a.bundle/'manifest.json']", "files += [Path(__file__),ROOT/'scripts/check_early_support_dispatch_v1.py',\n"
         "        ROOT/'scripts/native_early_support_tail_v1.py',archive,a.bundle/'manifest.json']")]
    for before,after in replacements:
        if audit.count(before)!=1:raise ValueError('Changed audit contract '+before)
        audit=audit.replace(before,after)
    ast.parse(text);ast.parse(audit)
    output.mkdir(parents=True)
    entry=output/'evaluate_early_support.py';entry.write_text(text)
    audit_path=output/'audit_early_support.py';audit_path.write_text(audit)
    files=[source,audit_source,Path(__file__),root/'scripts/native_early_support_tail_v1.py',
        root/'scripts/check_early_support_dispatch_v1.py',root/'scripts/native_head_ground_observation_v1.py',
        root/'scripts/native_isolation_microtrace_v1.py',root/'scripts/native_output_reserve_v1.py']
    result=dict(entry=str(entry),entry_sha256=sha(entry),audit=str(audit_path),audit_sha256=sha(audit_path),
        source_sha256={str(p.resolve()):sha(p) for p in files},native_execution=False,
        online_mpc=False,feedback_recomputed_from_actual_state=True,nominal_trajectory_only=False,
        head_observer=True,unchanged_lane_full_episode_check=True,scoring_changed=False,
        strict_tolerances_changed=False,valid_expert_labels=False)
    (output/'preparation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(prepare(a.root,a.output)))
