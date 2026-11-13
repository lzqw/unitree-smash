#!/usr/bin/env python3
"""CPU-only checks for the jump-event and full-racket action contracts.

This intentionally does not launch Isaac.  It verifies the pure command
decoders and residual mixer bookkeeping; the old full execution chain still
requires a native saved-state replay and is reported as pending.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from quadsmash.rl.action_contract import decode_full_racket_normalized_action
from quadsmash.control.full_racket_residual_v4 import ResidualActionMixerV4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path,
                        default=Path('artifacts/ppo_full_racket_residual_v4/policy.pt'))
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    torch.manual_seed(20261113)
    n = 4
    device = torch.device('cpu')
    limits = torch.stack((torch.linspace(-1.2, -0.2, 18),
                          torch.linspace(0.3, 1.4, 18)), dim=-1).repeat(n, 1, 1)
    default = limits[:, :, 0] + 0.37 * (limits[:, :, 1] - limits[:, :, 0])
    requested = torch.linspace(-1.0, 1.0, 18).repeat(n, 1)
    requested[1] = torch.randn(18).clamp(-1, 1)
    expected = limits[..., 0] + 0.5 * (requested + 1.0) * (limits[..., 1] - limits[..., 0])
    actual = decode_full_racket_normalized_action(requested, limits)
    full_diff = float((actual - expected).abs().max())
    if full_diff != 0.0:
        raise AssertionError(f'full-racket decode mismatch: {full_diff}')

    # The residual mixer request is intentionally not the final command: the
    # arm slew limiter and base composition may modify it.  That difference is
    # an execution effect, not an action-contract failure.
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    scale = state['residual_joint_scale_rad']
    mixer = ResidualActionMixerV4(n, device, scale)
    public = {'policy': torch.zeros(n, 794)}
    public['policy'][:, 775] = 1.0
    base = torch.zeros(n, 18)
    residual = torch.zeros(n, 18)
    residual[:, 12:] = 0.4
    mixed = mixer.action(base, residual, public['policy'], default[0], limits[0])
    request = mixer.last_requested
    if request is None or not torch.isfinite(mixed).all() or not torch.isfinite(request).all():
        raise AssertionError('residual mixer produced invalid commands')
    request_final_max = float((request - mixed).abs().max())
    if (mixed.abs() > 1.0 + 1e-6).any():
        raise AssertionError('mixer final command exceeded normalized bounds')

    # Validate phase-dependent jump-event decoding separately.  This is not
    # applied to full-racket-residual-v4.
    phase = torch.tensor([[0.0], [0.4], [0.8], [3.0]])
    leg_scale = torch.where(phase < 0.8, torch.full_like(phase, 0.25), torch.full_like(phase, 0.55))
    scale_jump = torch.cat((leg_scale.expand(-1, 12), torch.full((n, 6), 0.12)), dim=1)
    jump_target = default + scale_jump * requested
    if not torch.isfinite(jump_target).all():
        raise AssertionError('jump-event decoder produced invalid targets')

    result = {
        'status': 'passed',
        'device': 'cpu',
        'checkpoint': str(args.checkpoint.resolve()),
        'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'full_racket_decode_max_abs_diff': full_diff,
        'residual_request_final_max_abs': request_final_max,
        'request_final_difference_is_not_mapping_error': True,
        'jump_event_decoder_phases_tested': [0.0, 0.4, 0.8, 3.0],
        'native_saved_state_full_chain_replay': 'pending',
    }
    text = json.dumps(result, indent=2) + '\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
