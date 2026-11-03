#!/usr/bin/env python3
"""Build complete5 retention from finished native traces and command replay.

This reads the actual completed evaluation, checks all 128 first episodes and
replays every command of accepted full episodes through the production
controller. It is offline data validation, not another physical evaluation.
"""
import argparse
import ast
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quadsmash.rl.episode_trace_v4 import unpack_public
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases, summarize_training
from scripts.export_complete_demonstrations_v1 import classify
from quadsmash.control.full_racket_body_residual_v4 import ResidualActionMixerV4
from quadsmash.control.shared_target_assistance_v6 import LandingAssistedController
from quadsmash.control.persistent_recovery_assistance_v1 import PersistentRecoveryAssistanceV1
from quadsmash.control.support_policy_contract_v1 import policy_support_config, add_support_controller

SHARED_SCHEMA = 'g07-public-shared-target-bias-v5'
RESIDUAL_SCHEMA = 'g07-shared-target-residual-ppo-v1'


def selected_definitions(path,names,namespace):
    # Read the production controller without executing its Isaac launcher.
    tree=ast.parse(path.read_text(),filename=str(path))
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError('Production controller definitions missing')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation', type=Path, required=True)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous retention datasets')
    torch.set_num_threads(1)
    source = args.evaluation.resolve()
    protocol_path = ROOT/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'
    protocol = json.loads(protocol_path.read_text()); protocol['_root'] = str(ROOT)
    inputs = json.loads((source/'INPUTS.json').read_text())
    completed = json.loads((source/'completed.json').read_text())
    rows = json.loads((source/'episodes.json').read_text())
    metrics = json.loads((source/'metrics.json').read_text())
    admission = json.loads((source/'native_admission.json').read_text())
    cases, indices = training_cases(ROOT, protocol)
    summary = summarize_training(rows, cases, indices, protocol)
    if (inputs['policy_sha256'] != sha(args.policy) or inputs['protocol_sha256'] != sha(protocol_path)
            or inputs['training_indices'] != indices or any(inputs['reset_noise_amplitudes'].values())
            or summary['numerical_failures'] or not admission['passed']
            or completed['env_steps'] != 102400 or metrics['actual_control_env_steps'] != 102400
            or (source/'error.txt').exists()
            or any(metrics.get(k) != v for k, v in summary.items())):
        raise ValueError('Native evaluation identity, completion or independent counts differ')
    policy = torch.load(args.policy, map_location='cpu', weights_only=False)
    trace_root = source/'fixed_policy_traces'
    trace_manifest = trace_root/'manifest.json'
    manifest = json.loads(trace_manifest.read_text())
    if (len(manifest['episodes']) != 128
            or manifest['provenance']['policy_sha256'] != sha(args.policy)
            or manifest['provenance']['protocol_sha256'] != sha(protocol_path)
            or inputs.get('recovery_assistance_execution') != 'reset-persistent-recovery-assistance-v1'):
        raise ValueError('Require all original128 native persistent-recovery traces')
    row_by_index = {r['source_training_index']: r for r in rows}
    selected, observations, dispatched = [], [], []
    for entry in manifest['episodes']:
        index = entry['source_training_index']
        row = row_by_index[index]
        if entry['raw_episode'] != row:
            raise ValueError('Trace episode differs from actual evaluation row')
        if not classify(row, protocol, 5.)['curriculum_complete']:
            continue
        path = trace_root/entry['data_file']
        if sha(path) != entry['data_sha256']:
            raise ValueError('Trace data changed')
        with np.load(path, allow_pickle=False) as archive:
            public = unpack_public(archive['packed_public'])
            command = archive['dispatched_actions'].copy()
        if (public.shape != (800, 794) or command.shape != (800, 18)
                or hashlib.sha256(public.tobytes()).hexdigest() != entry['full_public794_sha256']
                or not np.isfinite(command).all() or np.abs(command).max() > 1.000001):
            raise ValueError('Require an intact full8s successful episode')
        selected.append(entry); observations.append(public); dispatched.append(command)
    if not selected:
        raise ValueError('No full complete5 episodes')
    production = ROOT/'scripts/evaluate_shared_target_ground64_persistent_recovery_v1.py'
    selected_definitions(production, ['load_shared_factory','SharedTargetResidualController'], globals())
    controller = SharedTargetResidualController(policy, len(selected), 'cpu')
    controller.reset()
    prov = manifest['provenance']
    default = torch.tensor(prov['default_joint_position_rad'])
    limits = torch.tensor(prov['joint_limits_rad'])
    frames = np.stack(observations).transpose(1, 0, 2)
    commands = np.stack(dispatched).transpose(1, 0, 2)
    targets = np.empty_like(commands)
    errors = np.zeros(len(selected))
    with torch.inference_mode():
        for tick, frame in enumerate(frames):
            raw = torch.from_numpy(frame.copy())
            targets[tick] = controller.actor(raw).numpy()
            command = controller.action(TensorDict({'policy':raw}, batch_size=[len(raw)]),default,limits).numpy()
            errors = np.maximum(errors, np.abs(command-commands[tick]).max(axis=1))
    if not np.isfinite(targets).all() or errors.max() > 5e-4:
        raise ValueError('Full-command replay mismatch: '+str(errors.tolist()))
    args.output.mkdir(parents=True)
    data_path = args.output/'residual_retention.npz'
    np.savez_compressed(data_path, observations=np.concatenate(observations),
        residual_targets=targets.transpose(1,0,2).reshape(-1,18),
        dispatched_actions=np.concatenate(dispatched), episode_offsets=np.arange(len(selected)+1)*800)
    evidence_names=['INPUTS.json','episodes.json','metrics.json','completed.json','execution_source_hashes.json']
    sources=[dict(policy=str(args.policy.resolve()),policy_sha256=sha(args.policy),
        metadata=policy['residual_training_metadata'],
        residual_joint_scale_rad=policy['residual_joint_scale_rad'],
        residual_arm_target_rate_rad_s=policy['residual_arm_target_rate_rad_s'],
        effective_reference_parameters_after_execution=controller.reference_controller.parameters[0].tolist(),
        command_replay_validated=True,source_evaluation=str(source),
        source_evidence={name:sha(source/name) for name in evidence_names},
        trace_manifest_sha256=sha(trace_manifest))]
    episodes=[dict(source_index=0,source_training_index=e['source_training_index'],
        case_id=e['case_id'],samples=800,max_command_error=float(errors[i]),
        source_trace_sha256=e['data_sha256'],command_replay_validated=True)
        for i,e in enumerate(selected)]
    result=dict(schema='verified-fixed-success-residual-retention-v1',sources=sources,episodes=episodes,
        episode_count=len(selected),samples=len(selected)*800,protocol_sha256=sha(protocol_path),
        minimum_speed_mps=5.,target_interface='raw-residual18-before-mixer',
        data_file=data_path.name,data_sha256=sha(data_path),cpu_gpu_command_tolerance=5e-4,
        label_validation='actor-and-command-replay-v1',action_override_at_deployment=False,
        production_controller_sha256=sha(production),exporter_sha256=sha(__file__),formal_acceptance=False,
        scope='Offline source-actor targets and full-command replay of actual finished128 native episodes.')
    (args.output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(episodes=len(selected),samples=len(selected)*800,
        max_command_error=float(errors.max()),manifest_sha256=sha(args.output/'manifest.json'))))


if __name__ == '__main__':
    main()
