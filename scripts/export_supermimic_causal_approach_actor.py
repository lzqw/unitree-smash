"""Export the phase-one residual policy with its recorded causal gate contract."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from quadsmash.rl.frozen_v2_approach_v1 import CausalApproachActorExport
from quadsmash.rl.supermimic_physical_loop_loading_v1 import tensor_digest
from quadsmash.rl.supermimic_policy_execution_v1 import POSTHIT, contract, resolve_contract
from quadsmash.rl.supermimic_terminal_scoring import approach_ball_scoring_protocol
from quadsmash.rl.supermimic_v1 import load_ppo_initialized_actor


def export_checkpoint(checkpoint, output, *, initial_gate_seconds=None):
    state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    if state.get('algorithm') != 'supermimic_v1':
        raise ValueError('Expected a SuperMimic checkpoint')
    metadata = state['metadata']
    execution, _ = resolve_contract(state)
    if execution['recovery_gate'] != POSTHIT:
        raise ValueError('The frozen Actor must use the causal post-release Recovery gate')
    if initial_gate_seconds is not None:
        if 'approach_ball_scoring' in metadata['task_contract']:
            raise ValueError('A trained ball-side checkpoint must retain its gate')
        execution = contract(arm_preparation_seconds=initial_gate_seconds)
    source = metadata['init_policy_checkpoint']
    actor, loaded = load_ppo_initialized_actor(Path(source['path']), torch.device('cpu'))
    if loaded['sha256'] != source['sha256']:
        raise ValueError('Initialization source changed')
    actor.load_state_dict(state['actor'], strict=True)
    actor.configure_approach_gate(execution.get('arm_preparation_seconds'))
    module = torch.jit.freeze(torch.jit.script(CausalApproachActorExport(actor).eval()),
                              preserved_attrs=['action_gate'])
    with checkpoint.open('rb') as stream:
        source_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
    task = dict(metadata['task_contract'])
    if 'approach_ball_scoring' not in task:
        task.pop('relaxed_scoring', None)
        task['approach_ball_scoring'] = approach_ball_scoring_protocol()
    contract_json = json.dumps(dict(
        schema='supermimic-causal-approach-request-v1', policy_execution=execution,
        inputs=['observation', 'had_contact', 'released'], initial_actor_sha256=source['sha256'],
        checkpoint_sha256=source_sha256, actor_state_sha256=tensor_digest(actor.state_dict().items()),
        task_contract=task), sort_keys=True)
    tmp = output.with_suffix(output.suffix + '.tmp')
    torch.jit.save(module, str(tmp), _extra_files={'contract.json': contract_json})
    tmp.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--initial-gate-seconds', type=float, help='E1 A0 override for historical weights')
    args = parser.parse_args()
    export_checkpoint(args.checkpoint, args.output, initial_gate_seconds=args.initial_gate_seconds)


if __name__ == '__main__':
    main()
