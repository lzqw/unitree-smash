# SuperMimic A0/A1/A2 implementation report

## Scope

This report documents the fixed-case paired evaluation, Recovery block refresh,
and Approach TOI-racket-only objective implemented on branch
`feat/supermimic-hybrid-toi-v13-20261119`.

## A0 fixed-case paired evaluation

`--paired-evaluation-baseline` accepts one JSON list containing exactly 128
unique `case_id` records. Every frozen refresh compares the same case IDs and
writes a report under `paired_evaluation/`.

Each report contains contact, ball-side, conditional recovery, Complete5 and
native-return summaries, per-case return and failure transitions, Actor
parameter delta, deterministic action RMS, Gaussian Actor KL when log standard
deviations are available, Critic TD/RTG error, and WM holdout quality.

The baseline file is copied into `paired_evaluation/pre-update.json`, so the
comparison remains auditable without relying on an external file after the run.

Critic and WM diagnostics are written after the frozen refresh has produced its
current native RTG and WM quality results. A missing calibrated WM bound is
reported as unverified and is never converted into a pass.

## A1 Recovery scheduling

`--refresh-mode per-head --head-block-size 8` gives each Recovery block an
independent frozen 128-case refresh. The same mechanism supports successive
R8 blocks and records each `post-Recovery` paired snapshot. Recovery reference
tracking is enabled only when the reference bank geometry, execution contract,
source identity, and holdout-lane exclusions match the current run.

Reference banks with an unknown or mismatched execution contract remain usable
for validation with zero tracking weight, but cannot silently provide training
tracking.

## A2 Approach TOI-racket-only objective

`--approach-objective toi_racket_only` restricts the Approach auxiliary loss to
an event-aligned time-of-impact neighborhood. It compares differentiable
racket task-space features containing racket position, face normal, linear and
angular velocity, root motion, and ball-relative geometry. Contact and ball
impulse gradients remain detached when used with `--training-stage
precontact_only` or `shadow_policy`.

The current implementation uses verified event-aligned reference anchors. It
does not claim that a new analytic free-flight root solver or PhysX contact
gradient has been validated. That validation remains a separate gate.

## Previous canary evidence

The earlier R8→A8 canary completed 8 accepted Recovery updates and 8 accepted
Approach updates, with one Approach backtrack. Its fixed refresh produced
73/128 qualified contacts, 50/128 ball-side completions, 38/128 conditional
Recovery successes, 25/128 Complete5, and mean native return -10.7314. The
initial reference was 74/128, 54/128, 26/128, and -10.8740 respectively.
This run predates the complete A0 report path and used detached contact BPTT,
so it does not establish a learned improvement.

## Verification

The relevant test set passes 34 tests. No new Isaac training run is claimed by
this code change.

