# SuperMimic world-model candidate gate

## Scope

This report records a frozen recorded-command comparison. It does not run
Isaac, does not train an actor, and does not certify a PhysX action response.

The anchor uses the original physical and legacy ball checkpoints together
with the fitted causal mode action-ball candidate. The calibrated candidate is
the 64-update contact-window WM checkpoint. Both are evaluated on the fixed
validation lanes from `validation_split.json` and the same event-aligned
windows.

## Evidence

| item | value |
|---|---|
| packed data SHA256 | `23a2a79dd902315f329a7a03f18c2cf19d6baee52634c30d838182133662a071` |
| action-ball candidate SHA256 | `12a72c6d2ee2f64d8482baec75d2e0e5a93641e65fb24bc701e8d35b1c5d5894` |
| calibrated WM checkpoint SHA256 | `6b13dde614f0391a62bcf53394634c6b15c274ce10a0e75cd34a0b56f24e1ac9` |
| validation output | `/home/autodrive/LZQW/experiments/supermimic_wm_validation_original_anchor_vs_contact_calibration_20261118_184315825265307` |
| promotion decision | `/home/autodrive/LZQW/experiments/promotion_decision_contact_calibration_20261118_184315825265307.json` |
| native simulation | `false` |

## Held-out contact-edge comparison

Lower is better.

| metric | anchor | calibrated candidate | result |
|---|---:|---:|---|
| contact edge, t=0 velocity RMSE | 1.7653 | 3.2313 | reject |
| contact edge, t=0 forward-sign error | 0.2609 | 0.5217 | reject |
| one tick before velocity RMSE | 5.1692 | 3.1667 | improved |
| one tick after velocity RMSE | 0.0818 | 0.0306 | improved |
| t=0 to t=4 velocity RMSE | 1.7221 | 2.2561 | reject |

The candidate improves pre-impact and post-impact metrics while degrading the
actual contact edge. It is therefore not promoted. The historical anchor and
the candidate checkpoint remain preserved separately.

## Code contract added

`supermimic_wm_candidate_gate_v1.py` compares the same held-out windows and
rejects missing, non-finite, or degraded protected metrics. The contact-edge
velocity metric must improve before promotion. The training script now saves a
`wm_calibration_candidate.pt` artifact and writes a promotion-required record,
instead of treating a model-loss run as an accepted WM.

CPU contract tests for the gate, replay, action-conditioned ball, and large-run
contracts passed with `23 passed`.

## Next required evidence

The model-only check does not establish the sign or magnitude of the real
PhysX action response. A paired native action perturbation test is still
required before enabling `full_contact` actor updates.
