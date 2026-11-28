# Recovery request saturation and landing lift

The weak-tail run finished its fixed128 evaluation with 77 body contacts, 5 strict and 22 relaxed recoveries among 108 qualified contacts, 96 ball-side returns and zero robot-net faults. PPO01 remains the reference. A later H4 candidate is in native validation.

## Request and backward checks

We reconstructed the 18 raw actuator requests ten control ticks before first body contact on all 146 eligible baseline falls. The replay of the clipped request agreed with the recorded command within 1.8e-7. The left rear calf raw mean was −1.189, with base contribution −0.00032 and Recovery contribution −1.189. This joint clipped on 139/146 cases, and on 133/146 when replaying the weak-tail moderate candidate. The other frequently saturated legs were front right thigh (135), left rear thigh (74), right rear calf (74), front left hip (67), and front right calf (62). [Per-joint evidence](r13_fault10_requests.json)

The new BPTT option preserves the native clipped forward request. It passes upstream descent directions that point back into the box and suppresses outward directions beyond the boundary. This is a surrogate derivative, and a parameter update must still clear the boundary to change a saturated native command. During a four-tick CPU replay through G07, the physical world model, task reward and tail, each ensemble member supplied an inward gradient at 14/18 saturated left-rear-calf positions in the early group. The late group supplied an inward direction at 9, 9 and 12 of 28 positions. The diagnostic uses eight cases per group. These counts are lane-ticks and cannot establish native recovery improvement. [Backward evidence](r13_directed_clamp_gradients.json)

## New training block

The new option applies an extra landed penalty to leg requests outside ±0.95, emphasizing the left rear thigh and calf. Height reward rises linearly over 0.12–0.30 m, and leg-speed damping ramps up between 0.15 and 0.20 m. The airborne objective encourages smaller rear-to-front foot height difference and retains the existing attitude and angular velocity terms. Sampling starts at least 50 ms after release for airborne cases and reserves distinct cases in the first 24 ticks after landing, alongside late pre-fall starts. The 21-surface geometry, 75 mm rear-thigh target and 0.1 tail-gradient scale stay active.

The trainer supports 24/32-step differentiable segments. This first comparison uses four ticks from native anchors because the independently measured pre-fall action response loses directional accuracy at H16/H32. Longer continuous airborne segments still need a phase-specific action-response check. The next block is queued behind the current H4 official evaluation and selects its retained checkpoint. Neither a split head nor a new Approach distribution is part of this first change.

## Airborne response and actual motor targets

A follow-up replay compared an earlier Recovery candidate and PPO01 from 121 identical native prefixes. Of those, 13 holdout pairs were still airborne at H24. For the 12 cases with a measured change over 0.1 mm in rear-minus-front foot height, all three WM members predicted its sign correctly. The measured RMS was 0.39 mm and none exceeded 1 mm. At H32 eleven held baseline trajectories had landed; the foot-height response cosine fell to −.022 to .103 for this group. These measurements support testing longer windows for modest airborne changes while keeping short gradients near touchdown. [Measured response](r13_air_preparation_response.json)

The queued landing-assist block now uses 8, 16 or 24 ticks where the entire segment fits before measured first touchdown. Elsewhere it keeps four ticks. The future landing tick is used only to choose the training horizon. A replay of the PPO01 sampler planned 24 H24 segments, 248 H16, 226 H8 and 6,214 H4 over twenty updates. An end-to-end H24 backward check on six eligible native starts produced finite Recovery gradient norms of 11.00, 11.47 and 11.16. These are model-side measurements, pending native evaluation.

For the previous H4 candidates we restored the G07 controller snapshot, replayed the incumbent history and compared candidate commands at identical measured states ten ticks before first body contact. All 146 incumbent motor commands reproduced within 0.00000048 rad. Of 139 saturated left rear calves, the conservative candidate brought zero back inside the request range. The moderate candidate brought three inside, all three changing the final target by more than 0.001 rad. In the moderate candidate the count clearing saturation went from six after BPTT to five after success retention and three after pre-fall localization. The queued block will repeat the final motor-target audit for both candidates. [Motor-target evidence](r13_h4_fault10_motor_targets.json)

## Does leaving calf saturation prevent a fall?

In the R13 training partition, we compared each eligible first body contact with all clean successful episodes from the same panel and Approach at the same elapsed tick after first foot touchdown. Each failure received one vote, regardless of how many matched controls it had. At ten ticks before the fall, 110 failures matched 41 distinct success cases. Lower-bound calf requests occurred on 94.5% of failures and 68.9% of the weighted controls. Mean root heights were similar (158 versus 156 mm), while left rear thigh clearance differed (59 versus 120 mm). Only 12 held failures had a matched success control, using six distinct controls. These are observational comparisons on native states.

A separate, completed 16-tick native intervention started 32 ticks before first body contact. On 67 of 145 exact paired falls, a previously clipped left rear calf request came inside the box and the commanded joint target extended by more than 0.001 rad on at least one such tick. The median per-case peak extension on these ticks was 0.0105 rad. Of these 67 falls, 35 occurred later, 18 at the same tick and 14 earlier; median delay was one control tick. All 67 still fell. Multiple joints changed simultaneously. This historical intervention used the earlier moderate candidate; the directed-lift policy still requires its own native outcomes.

As a model-only check, we isolated the left rear calf command sequence from the other 17 commands in the same windows. We averaged the calf contribution across both insertion orders. Among 124 pairs with more than 0.001 rad calf target change by H4, the projection of the calf component onto the model's full thigh-clearance response was .055–.085 across members. This estimates a contribution within the WM and cannot assign a causal share of native falls. [Paired control and intervention data](r13_native_calf_controls.json)

## Preceding H4 block on native64

Both candidates finished deterministic native evaluation on the same 64 launches. Each candidate was paired to PPO01 in its own simulator lane positions. All 54 qualified contacts and all 46 ball-side successes retained their case identities in both slots. Neither candidate incurred a robot net fault.

| Candidate | Body contact / 54 | Strict / 54 | Relaxed / 54 |
|---|---:|---:|---:|
| Conservative | 31 → 30 | 3 → 4 | 9 → 11 |
| Moderate | 32 → 32 | 3 → 6 | 8 → 12 |

The conservative candidate avoided three baseline body contacts and incurred two new ones. Its lower body count selected it for the official fixed128 run. The moderate candidate avoided three and incurred three. This experiment used the preceding H4 objective. The directed-clamp and height-ramp block is queued behind fixed128. [Paired case flips and checkpoint identities](r13_h4_native64_flips.json)

## Preserving early corrections through the final distillation

The final localization step previously trained the student to reproduce PPO01 until a 12-tick sustained warning fired. On the training partition, the warning was inactive on all 2,775 eligible airborne ticks of 110 episodes with later measured body contact, and on 2,480 of their 2,640 initial-support ticks. This defeated the airborne preparation objective at the final distillation step.

With landing assistance enabled, the student now receives the teacher correction on these native airborne and initial-support windows from 50 ms after release. Initial support comprises the first 24 states after measured touchdown and ends early on a body fault. The loss still anchors clean successes and anchors early leg commands during the 40 ms release protection. The future fall label selects training targets only.

A CPU replay held the post-retention H4 teacher fixed and changed only the localization procedure. Across 19 held fall cases, the projection of the conservative student's raw head correction onto the teacher correction increased from −0.073 to 0.873 in air and from 0.349 to 0.908 during initial support. The corresponding moderate projections were −0.048 → 0.838 and 0.341 → 0.869. Final clipped request RMS stayed within the two action budgets. These are teacher-fidelity measurements at recorded states. The running H4 native evaluation uses the original localization; the queued directed-lift block uses the repaired one. [Fitting coverage and paired changes](r13_landing_teacher_coverage.json)
