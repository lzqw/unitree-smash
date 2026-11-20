# SuperMimic small GPU canary

## Run

- Branch: `feat/supermimic-hybrid-toi-v13-20261119`
- Run directory: `supermimic_canary_20261120_r8a8r8_v11`
- Profile: `shadow_policy`
- Contact BPTT: `detached`
- Schedule: `R8 -> A8 -> combined frozen refresh`
- Formal evaluation: not run

## Actor and model updates

- Recovery effective updates: `8/8`
- Approach effective updates: `8/8`
- Proposal attempts: `16`
- Rejected proposals: `0`
- Backtracked proposals: `1` in Approach
- Shared WM updates: `66` total, including `2` refresh updates
- Critic updates: `1003` total, including `1` refresh calibration update
- Base actor remained frozen during actor updates
- Shared WM remained unchanged during actor updates
- Tracking was disabled because no reference bank was supplied

## Frozen combined refresh

- Complete lanes: `128/128`
- Collection: `800` control ticks and `102400` transitions
- Case identities: `128` unique IDs from the task suite
- Actor identity remained unchanged during collection and refresh
- Frozen return monitor: no decline or plateau stop

Observed native metrics for this diagnostic refresh were:

| Metric | Result |
|---|---:|
| Qualified airborne string contact | 73/128 (57.03%) |
| Ball-side complete before recovery | 50/128 (39.06%) |
| Conditional recovery after qualified contact | 38/73 (52.05%) |
| Complete5 | 25/128 (19.53%) |
| Mean native return | -10.7314 |

These values are frozen training-refresh diagnostics. They are not formal promotion results.

## WM refresh audit

Both heads consumed the new online stream together with initial and older data. The Recovery WM batch had no source or event shortfall. The Approach batch had expected event scarcity, with fallback sampling and shortfalls for `contact_release`, `approach_failure`, and some older/initial sources. The refresh completed and was written to the run manifest.

## Engineering fixes verified during the run

The run exposed and fixed three closure defects.

1. Mutable actor-start masks were allocated under inference mode. They are now allocated as ordinary CPU tensors before deferred terminal labeling.
2. Frozen terminal records did not preserve stable task `case_id`. Case identity now flows from the task suite through native scoring and return monitoring.
3. `refresh_world_budgets` was imported only in `main()` and was unavailable to the runtime scheduler. It is now imported in the runner scope.

Targeted regression tests passed after these fixes. The run completed with `training_completed.json`, `cycle_01.json`, the frozen refresh manifest, and the final checkpoint.

This canary validates the software data/optimizer/refresh path. It does not validate the unapproved contact Jacobian or justify unattended large-scale training.
