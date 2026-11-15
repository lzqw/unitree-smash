# SuperMimic refresh-only critic calibration: finite validation

## Scope

- Code commit: `b12b7a5`
- Source policy: `frozen_v2_smooth_approach16_env5120_01/policy.pt`
- Source SHA256: `8f510fe9f218beef791ac21675a19ed25d11f510a83fe247cddf2893bee08399`
- Candidate checkpoint: `candidate_0002.pt`
- Candidate SHA256: `c6fbf8f09b416d69fc38b06c3a7a33c721f3ee5e3eed0b737325b7d1c22fa071`
- GPU: `cuda:0`, GPU dynamics enabled, 128 environments, 800 control ticks
- No MPC, teacher, video, or additional evaluator intervention

The refresh critic now fits frozen-policy native RTG tensors directly. The previous
refresh implementation computed `rtg_refresh` but trained the critic on an imagined
rollout instead. The run also used refresh-only archive persistence and did not change
the task, reward, action mapping, physics, or imagination/world-model horizons.

## Finite training result

- Recovery proposals: 2; accepted: 0
- Approach proposals: 2; accepted: 2
- Rejected/backtracked: 0/0
- Refresh ticks: 800 (`102400` transitions)
- Refresh world updates: 8
- Refresh native-RTG critic updates: 8
- Total world updates: 10
- Total critic updates: 430
- Final actor updates: 2
- Native mixed-policy monitoring return remained `-12.526784349736634` during the
  two short post-refresh collection windows; this is monitoring data, not the formal
  frozen evaluation.

## Formal 128-environment evaluation

| Metric | π0 | Candidate | Change |
|---|---:|---:|---:|
| Undiscounted native return | -12.52678435 | **-12.30243368** | **+0.22435067** |
| Qualified airborne string contact | 74/128 | 73/128 | -1 |
| Ball-side complete except recovery | 53/128 | **54/128** | +1 |
| Conditional recovery | 34/74 | 34/73 | denominator changed with contact |
| complete5 | 27/128 | **25/128** | -2 |
| Numerical failures | 0 | 0 | 0 |

Independent head/all-body contact labels were unavailable; those fields remain
unknown rather than being treated as false.

The candidate passes the native-return and ball-side thresholds, but fails the
complete5 threshold (`25 < 27`). It is therefore rejected and is not promoted to
`native_accepted_reference`; no large-scale training follows this run.

## Set changes versus π0

- Ball-side lost: `A-720401-018`, `A-720401-032`
- Ball-side added: `A-720401-026`, `A-720401-040`, `B-720402-018`
- Recovery lost: `A-720401-027`, `A-720401-035`, `A-720401-037`, `B-720402-008`
- Recovery added: `A-720401-022`, `A-720401-061`, `B-720402-001`, `B-720402-013`
- complete5 lost: `A-720401-027`, `A-720401-032`, `A-720401-035`, `A-720401-037`
- complete5 added: `A-720401-061`, `B-720402-013`

The positive return change is therefore not a clean recovery-only gain: some severe
negative episodes improved, while four complete successes were lost and only two
new complete successes appeared.

## Decision

This is a useful diagnostic improvement in native return and ball-side coverage, but
not an accepted task improvement because complete5 regressed. Keep the checkpoint and
all refresh data for diagnosis; stop before any larger training or parameter scan.

Artifacts:

- Training: `/tmp/supermimic_physical_loop_capture_20261114_ub0eiq8t/refresh_fix_r2_rerun`
- Evaluation: `/tmp/supermimic_physical_loop_capture_20261114_ub0eiq8t/eval_refresh_fix_r2_rerun`
