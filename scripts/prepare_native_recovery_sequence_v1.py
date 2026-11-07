"""Prepare an isolated native sequence diagnostic without changing live entry."""
import argparse
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def transform(source,root):
    def once(old,new):
        nonlocal source
        if source.count(old)!=1:raise ValueError('Native entry changed: '+old[:100])
        source=source.replace(old,new,1)
    once('ROOT = Path(__file__).resolve().parents[1]',f'ROOT = Path({str(root)!r})')
    once("parser.add_argument('--checkpoint', type=Path, required=True)",
        "parser.add_argument('--checkpoint', type=Path, required=True)\n"
        "parser.add_argument('--recovery-sequence-bundle', type=Path, required=True)\n"
        "parser.add_argument('--recovery-sequence-case', type=int, required=True)")
    once("    learned_recovery_checkpoint = None\n",
        "    if (not args.allow_declared_controller_variant or args.learned_recovery_command_head is not None\n"
        "            or args.landing_mpc_cases is not None or args.reactive_vhsip_cases is not None\n"
        "            or args.expert_bank is not None or args.landing_config is not None\n"
        "            or args.support_config is not None or args.disable_support):\n"
        "        raise ValueError('Sequence diagnostic requires exact original student and no other new teacher')\n"
        "    learned_recovery_checkpoint = None\n")
    once("        expert_bank=str(args.expert_bank.resolve()) if args.expert_bank else None,",
        "        recovery_sequence_bundle=str(args.recovery_sequence_bundle.resolve()),\n"
        "        recovery_sequence_case=args.recovery_sequence_case,\n"
        "        case_time_conditioned_diagnostic=True,\n"
        "        expert_bank=str(args.expert_bank.resolve()) if args.expert_bank else None,")
    once("    execution_snapshot=snapshot_execution_sources(ROOT,args.output)\n",
        "    execution_snapshot=snapshot_execution_sources(ROOT,args.output)\n"
        "    (args.output/'sequence_evaluator.py').write_text(Path(__file__).read_text())\n"
        "    atomic_json(args.output/'sequence_entry_provenance.json',dict(\n"
        "        path=str(Path(__file__).resolve()),sha256=digest(Path(__file__))))\n")
    once("        learned_recovery = None\n",
        "        from scripts.native_recovery_sequence_replay_v1 import NativeRecoverySequenceReplay\n"
        "        native_replay=NativeRecoverySequenceReplay(args.recovery_sequence_bundle,\n"
        "            args.recovery_sequence_case,limits,digest(args.checkpoint),args.output)\n"
        "        sequence_student_commands=[];sequence_final_commands=[];sequence_first_episode=[]\n"
        "        learned_recovery = None\n")
    once("                if action.shape != (count, 18) or not torch.isfinite(action).all() or (abs(action)>1).any():",
        "                replay_state=None\n"
        "                if tick>=native_replay.start and not bool(finished[native_replay.case]):\n"
        "                    replay_state=native_replay.model.state_from_native(env,native_replay.case)\n"
        "                sequence_student_commands.append(action.cpu().numpy().copy())\n"
        "                action=native_replay.action(tick,obs,action,finished,replay_state)\n"
        "                sequence_final_commands.append(action.cpu().numpy().copy())\n"
        "                sequence_first_episode.append((~finished).cpu().numpy().copy())\n"
        "                if action.shape != (count, 18) or not torch.isfinite(action).all() or (abs(action)>1).any():")
    once("        contact_observer.save()\n",
        "        contact_observer.save()\n"
        "        np.savez_compressed(args.output/'sequence_dispatch.npz',\n"
        "            student=np.stack(sequence_student_commands),final=np.stack(sequence_final_commands),\n"
        "            first_episode=np.stack(sequence_first_episode))\n"
        "        sequence_result=native_replay.save()\n")
    once('        summary["landing_blas_runtime"] = LANDING_BLAS_RUNTIME',
        '        summary["landing_blas_runtime"] = LANDING_BLAS_RUNTIME\n'
        '        summary["recovery_sequence_replay"] = dict(\n'
        '            case=native_replay.case,applied_steps=native_replay.applied,\n'
        '            query_matched=sequence_result["query_match"]["passed"],\n'
        '            deployable_risk_trigger=False,sustained_mpc_feedback_implemented=False,\n'
        '            result_path="sequence_native_result.json")')
    compile(source,'<prepared-native-sequence>','exec')
    return source


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve previous preparation')
    old=a.source.read_text();new=transform(old,ROOT);a.output.mkdir(parents=True)
    entry=a.output/'evaluate_sequence.py';entry.write_text(new)
    record=dict(live_entry_modified=False,source=str(a.source.resolve()),
        source_sha256=hashlib.sha256(old.encode()).hexdigest(),entry=str(entry.resolve()),
        entry_sha256=hashlib.sha256(new.encode()).hexdigest(),
        scope='Case/time-matched model execution diagnostic, not a sustained MPC or policy learning result.')
    (a.output/'preparation.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))

if __name__=='__main__':main()
