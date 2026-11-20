# Fixed-case policy and world model audit

The historical R8/A8 candidate v16 completed 25 of 128 cases. The initialization completed 26. Two cases became complete and three lost completion. Two runs of the same v16 weights yielded identical returns in every case. The subsequent per-head refresh reused those weights and its existing proposal counters. Its two refresh names do not establish an additional R8 or A8.

| Same 128 cases | Initialization | v16 |
|---|---:|---:|
| Qualified airborne contact | 74 | 73 |
| Ball-side completion | 54 | 50 |
| Complete5 | 26 | 25 |
| Mean native return | −10.8740 | −10.7314 |

The archived policy hashes and measured causal recovery latches identify both actors. On identical Approach-phase observations (512 rows), the normalized request RMS change was 0.000909 and the mean pre-clamp Gaussian KL per action dimension was 0.000167. On Recovery-phase observations (896 rows), the corresponding values were 0.001805 and 0.000657. These measurements compare the two composed policies. A separate post-Recovery/pre-Approach frozen evaluation is needed to attribute either change to one head.

## Critic

Each row uses 128 observed native rewards paired with the immediately preceding and following states in the same episode. Both critics were evaluated on the same rollout. The input encoder uses the matching checkpoint. The values describe the fixed training suite.

| Native step | TD MAE before refresh | TD MAE after refresh |
|---:|---:|---:|
| 0 | 0.05113 | 0.04683 |
| 256 | 0.11936 | 0.12055 |
| 400 | 0.06979 | 0.07128 |

## World model

On 320 shared windows under a second simulator seed and the historical case identities, 64-step ball velocity MAE fell from 0.5413 to 0.4473 m/s. Ball position error at the 95th percentile rose from 1.2948 to 1.3942 m. The online checkpoint's training history includes these case identities. A second evaluation uses 32 newly generated launch vectors disjoint from the training and formal test suites, with the checkpoints and launch vectors committed to a local manifest before simulation. Its manifest SHA256 is `543a168af3f5d5c94e566764a8972dbff3eb6066beb63083c91bb0ff78e82d8f`.

## Approach shadow and paired racket responses

Eight native cases ran to tick 256 with fixed model weights. A 64-step imagined Approach rollout matched eight verified references and included seven effective lane-steps at interception. The gradient norm from racket tracking to the Approach head was 2.65988, and the command preview matched the executed native command on all eight lanes (maximum absolute difference 0 rad). This run made zero optimizer updates and did not measure a change in task success. [Shadow results](toi_shadow_8.json)

On twelve held-command pairs from five probe specifications, we rolled out the two measured command sequences with each model and held the ball input common between the two branches. We measured racket motion by forward kinematics. At the second tick the online candidate had a median linear-velocity gain of 0.667 and direction cosine of 0.949. At the fourth tick these were 0.530 and 0.645. The fourth-tick angular-velocity gain was 0.371. These measured differences motivate continued action-response calibration before authorizing contact gradients. Absolute errors include all rows. Relative statistics use native differences greater than three times the repeated-nominal variation and the numerical floor. [Paired results and checkpoint hashes](paired_body_response.json)

## Contact transfer on the training candidate

With the same recorded commands, 32 cases, starting tick 240 and the calibrated analytic ball solver, the original robot model recovered 27 of 30 actual racket hits. The online v16 body recovered 15. Both produced one false positive in two no-hit cases. With v16 the racket-center error at the native hit had a 95th percentile of 264 mm, versus 87 mm with its parent. Resetting the robot state to the measured value each tick improved v16 detection to 30/30 and reduced the center error to 4.7 mm. This isolates accumulation of robot-state error as a contributor to missed hits. The online run may previously have encountered these case identities. Its results do not authorize contact gradients. [Paired rollout results](online_body_transfer64.json)

## Body model calibration with separate native data sources

A GPU run updated only the robot predictor 56 times at horizons 1, 2, 4, 8, 16, 32 and 64. Each batch selected ten natural windows, two contact windows and four one-step native command pairs. The split excluded the parent calibration's held-out lanes, the reference-bank exclusion list and the paired-probe validation lanes. Actor and Critic updates were both zero.

On the original 32-case sliding test, continuous robot rollouts recovered 28 of 30 native hits and produced zero hits on the two negative cases. At the native contact tick the 95th-percentile racket-center error was 68.5 mm. The parent recovered 27 hits, with 87.3 mm center error. The 64-step ball speed-vector error at the 95th percentile increased to 10.98 m/s from about 10.34 m/s with the parent. The sample contains only two negative cases. [Transfer results](mixed_body_transfer64.json)

On twelve separately held command pairs, the fourth-tick median linear velocity gain and direction cosine were 0.696 and 0.916. For angular velocity they were 0.535 and 0.962. The candidate checkpoint and fresh 32-case launch manifest hashes were recorded before the independent native capture finished. This candidate remains offline pending the independent evaluation and native action-gradient checks. [Paired measurements](mixed_body_paired.json) · [Prospective checkpoint binding](mixed_body_fresh_panel_binding.json)

The stored actor at the start of the historical Approach block was recovered from the original combined checkpoint. Exactly six Recovery parameters differ from the initialization and exactly six Approach parameters differ from the final pair. Its separately queued native evaluation uses the original 128 cases and seed 882072. [Weight identities](historical_r8_origin.json)

## Independent native forward panel and paired action response

A separate seed (882074) completed all 800 control ticks on 32 distinct native launches, yielding 25,600 transitions, 21 racket contact events, 320 uniformly sampled windows and 63 event-relative windows. The initial native process reported a Kit shutdown error after finishing cleanup. Offline rescoring validated the saved case identities, native arrays and complete windows. With the mixed robot predictor and the previous ball model, the one-step pre-contact velocity RMSE increased from 1.037 to 1.437 m/s and the four-step contact-centered error increased from 2.782 to 4.282 m/s. [Fresh panel](fresh32.json)

We then used the previously fitted B contact solver on the same data. All six event-window forward checks passed. In the predicted robot rollout the one-step contact velocity RMSE was 2.547 m/s and the four-step contact error was 0.203 m/s. Replacing predicted robot endpoints with measured ones reduced the one-step contact RMSE to 0.408 m/s. The solver choice was tested after inspecting the initial results. This comparison has diagnostic scope. [Predicted robot](solver_body.json) · [Measured robot](solver_teacher.json)

Twelve measured plus/minus command pairs gave eight cases with stable contact flags, interval counts and first-contact substep across all branches. Four changed first-contact substep. On the eight stable cases the midpoint Jacobian along the executed command sequence gave a median direction cosine of −0.183, tenth percentile −0.485 and median gain 0.040. Even when using the native incoming ball state and measured racket motion at all 160 event substeps, the median cosine was 0.036. Integrating the archived racket normal-force channel with incoming velocity and modeled flight forces gave a median cosine of 0.077. The residual is unattributed with the channels available in those traces. A favorable nominal outgoing velocity does not establish which action improves the return. No contact-gradient receipt was issued. [Paired response vectors and native force decomposition](solver_pairs_forces.json)

The native evaluator now records ordered implicit-drive parameters directly from the articulation. A separate CPU baseline used these parameters and causal command history to predict joint motion. At horizons 4 and 64, arm velocity MAE fell from 0.226 and 0.298 rad/s with executed targets to 0.118 and 0.156 rad/s with PD features and history. These figures evaluate joints under recorded commands. Deployment requires full-body and paired-response validation. [Actuator study](actuator_identification.json) · [Matched-state historical Actor changes](historical_r8_actions.json)

## Storage of causal G07 plan history

The training archive stores the full 794-coordinate public observations retained by the native planner as little-endian float64 matrices. Identical matrices can share immutable storage across before/after snapshots within a serialized chunk. Native probes retain their original recording format. Replay reconstructs independent Python lists before calling the original executor restoration routine.

We replayed a recorded control-boundary state on 128 lanes with the same GPU executor. All 2,304 final action components matched native execution exactly with either storage format. The reconstructed controller hashes also matched. Four independently captured copies of this one state occupied 80.24 MB in the previous representation and 10.56 MB with packed matrices, before zstd compression. These sizes measure executor memory at one tick, not total long-run disk usage. An initial CPU replay differed from the native GPU by up to 5.43e-6 in two command components. The GPU replay used the original 2e-6 comparison tolerance. [Native source and storage measurements](archive_codec_results.json)

## PD transfer to racket motion

We froze the 56-update Body WM, replaced only its six predicted arm joints with the independently calibrated PD predictor, and propagated the corrected arm state into subsequent steps. On the same twelve held action pairs the second-tick angular-velocity gain increased from 0.723 to 0.973, and the tenth-percentile directional cosine increased from −0.657 to 0.964. By the fourth tick, linear-velocity direction at the tenth percentile fell from 0.718 to 0.369. The initial hold age is known only from the recorded command prefix; the paired diagnostic initialized its lower bound at zero.

On another 32-lane held split the isolated PD joint model still had lower arm velocity MAE than the executed-target linear baseline. However, in 320 complete 64-step whole-body windows, replacing the arm increased racket position MAE from 32.6 to 51.4 mm, root angular velocity MAE from 0.151 to 0.184 rad/s, and full-body RMSE from 0.508 to 0.540. This replacement failed the non-regression requirement. The existing WM remains the policy-facing model. [Whole-body rollouts and paired action transfer](actuator_full_transfer.json) · [Joint forward evaluation](actuator_second_split.json)

## Recovery event windows and fixed-case attribution

On 32 held body-model lanes, Recovery windows included 19 distinct touchdown episodes, ten support-loss episodes, eight body-ground episodes and nine stable episodes. The mixed model reduced H64 angular-velocity RMSE on stable episodes from 0.0981 to 0.0639 rad/s. On touchdown and support loss it increased from 0.6628 to 0.6777 and from 0.6960 to 0.7070 rad/s respectively. These measurements do not establish independent task-head training provenance or calibrated acceptance bounds. [Recovery window measurements](recovery_wm_event_holdout.json)

The fixed-stage comparator checks full initial-state arrays, case-to-lane assignments, seed, physics, execution, independent scoring protocol, both Actor hashes, all 800 ticks and 128 terminal rows. An archived frozen run compared against itself yielded identical scores. A second real run with a different seed was rejected. The original post-Approach and post-refresh checkpoints both contain the same Actor state hash.

## Recovery tracking and delayed command response

A GPU preflight restored the original full 128-lane execution states at native ticks 384 and 448 and used the v16 Actor, Critic and WM. The reference bank contained 26 independently scored Recovery references. The training mask excluded 40 cases. With a tracking coefficient of 1 and 64 imagined steps, 76 and 74 lanes produced tracking gradients. Task gradient norms were 25.27 and 14.87, tracking gradient norms were 0.0582 and 0.0519, and their cosines were 0.214 and 0.196. All three model hashes were unchanged. This establishes a working Recovery tracking path at those two starts. [Measured gradients and weight hashes](recovery_tracking_preflight.json)

The offline robot fitter can now use both one-step measured interventions and multistep signed command sequences in its 25% identification allocation. It uses the common native start, both complete executed command prefixes, and predicted robot, ball and foot states at subsequent steps. Dividing the response error by the norm of the entire executed target difference preserves evidence after the target reconverges. The sequence loader checks identical native repetitions, complete source receipts, all four branches, shared physics and task contracts, lane exclusions and reset boundaries. A separate 16-step code-path check on eight measured training lanes ran the negative and nominal branches and obtained finite gradients. This check did not step an optimizer. The signed calibration job requires the completed positive branch. [Measured sequence preflight](sequence_preflight.json)
