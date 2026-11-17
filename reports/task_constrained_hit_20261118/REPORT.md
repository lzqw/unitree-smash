# Task-constrained joint impact search, baseline 910d6f2

Scope: offline teacher code and fixed CPU replay; independent historical PhysX calibration. No new-policy native evaluation, 128-case evaluation, training, or SuperMimic modification. Worktree `/home/autodrive/LZQW/unitree_smash_teacher_eval`.

## Implementation

New opt-in `--task-constrained-hit --anchored` replaces fixed-face rate retargeting. It cannot be combined with `--impact-feasible-rate`. Historical default behavior remains available.

1. `task_constrained_impact_goal_v2.py`: constrained max-min forward task margins. A solved inverse goal is distinct from a task-valid goal, robot-feasible contact, and executable fullbody trajectory. A 4.77m/s goal cannot pass regardless of optimizer convergence or boolean claims.
2. Downward attack height check: own-side contact must be above the lowest physical tape plus ball radius under the current zero-wind gravity/drag assumption. The check is necessary only. Actual crossing height/drop and target margins use the original forward flight model and court geometry.
3. `task_constrained_hit_search_v2.py`: simultaneously optimize six angles, six rates, two coordinates inside one authored string strip, and normalized worst task margin. Contact time is selected from the SAME original event offsets (24 events overall); no extra random samples or parameter sweep. Outgoing state comes from actual forward collision, not a free ideal variable. Task normal moves with the outgoing impulse direction. Limits remain 0.5mm fine position, 0.12 normal vector difference, original joint positions and 5rad/s rates.
4. Immutable anchors and original requested normal/velocity remain in diagnostics. Only fully valid snapshots are eligible for Hermite, full18D and actual terminal-impact validation. Final eligible candidates rank by actual task margin, not closeness to an invalid old ideal velocity. References still need final selected-plan admission and command consistency.
5. Fullbody solver failure now retains failed state, target, contact mask, preceding states and solver trace; failure is MODEL_UNCERTAIN, never a physical infeasibility certificate or execution permission.
6. Independent native calibration script replays historical actions from reset with no planner/controller networks. Analysis compares one-step measured-state CPU predictions against cumulative predictions. This does not validate any new teacher reference.

## Tests and fixed replay

Final CPU regression: **131 passed**, exit 0 (FINAL_TESTS.txt). Initial new test run had 1 failure because config validation is explicit; corrected the test to call `.validate()`, retaining NEW_TESTS.txt. Python compilation and source-only diff whitespace checks pass. Raw captured nvidia-smi/stdout/source excerpts retain trailing whitespace; the all-artifact diff check reports these formatting warnings, not source errors. Synthetic tests establish engineering contracts, not robot or PhysX performance.

Exactly one fixed CPU replay, exit 0: same source hashes, 8 planning calls, 24 event candidates. 10359 forward-impact calls in total; 8290 joint-optimizer queries across 12 fine-anchor events. These finite-difference queries are not separate episodes or a wider fixed-face sample search. Joint solve median 7.46s, p95 14.98s, max16.52s; offline compute time is NOT a rejection criterion or a real-time claim.

| Gate | Result |
|---|---:|
| Necessary downward height | 24/24 |
| Task-valid initial analytic goal | 20/24 |
| Initial position anchor <=3mm | 13/24 |
| Initial position anchor <=0.5mm | 12/24 |
| Joint contact: position + normal + bounded rate + analytic task | **2/24** |
| Pass full Hermite trajectory limits / enter18D | **1** |
| Full18D prediction pass | **0** |
| Exported qualified references | **0** |
| New reference PhysX / native hit / Recovery / complete5 | **NOT_RUN** |

The previous 910d6f2 fixed replay had zero task-valid rates and zero full18D calls. This is progress in analytic candidate generation, NOT native hit success or recovery improvement. Old position count14/24 versus current13/24 is also retained: new goal/UV/seed semantics differ, so do not call it a matched single-trajectory loss. The exact immutable Lane0 early anchor of about0.168mm is still present.

### Two task-valid contact states

| Event | Position | Normal-vector error | Predicted outgoing velocity (m/s) | Remaining failure |
|---|---:|---:|---|---|
| Lane0, planning2.49s, contact+.27s |0.48631mm|0.079809|[7.49796,1.05562,-4.08127]|Hermite interior rate exceeds original limit|
| Lane1, planning2.49s, contact+.29s |0.48371mm|0.079809|[5.05218,-1.97893,-1.32801]|18D node2 contact/motor active set fails|

The first contact has bounded endpoint rates but its full Hermite curve reaches about7.21745rad/s at node13/joint0 and exceeds the0.05rad/tick step limit by0.02210rad. It is rejected before fullbody prediction. The second passes trajectory joint position/rate/step checks, then fails the unchanged160-substep implicit model after two successful intervals. Neither is an executable reference.

Raw replay retains the initial pose's stale rejection label for Lane1, even though the new joint stage passed; CANDIDATE_STAGES.json derives the correct final rejection from its saved physical result. Post-replay diagnostic fixes make final/original fields explicit. No new candidate replay was run after these fixes. The final-margin ranking branch has no real fullbody success to exercise; production wiring tests cover it, not a performance claim.

## Takeoff requirement evidence

All24 predicted contact heights lie in1.67450–1.86466m, above the1.549m center necessary bound. Thus the minimal necessary height increase from this test is zero, NOT proof of sufficient net clearance or trustworthy long-horizon base prediction. Time/linear velocity/pitch/angle-rate corrections are recorded as unknown, unrealized targets; local failed solves cannot establish their global minima. Because two arm-level task states exist, there is no evidence here to implement a fabricated takeoff velocity or change WBC/base control.

## Native calibration

See CALIBRATION_SUMMARY.json and attempts/calibration01 for final outcome. It is a historical action replay, not a new candidate trial. Initial public state difference is0; physical dt remains62.5microseconds, control dt10ms, original two lanes [0,64], seed882072, original800ticks. Independent supervisor identity/cgroup and exit record are saved. All30 checked historical asset/protocol hashes match during run.

Torque telemetry caveat: raw `applied_drive_torque` comes from IsaacLab `ImplicitActuator.applied_effort` and is an APPROXIMATION, not measured PhysX drive torque. The pinned installed source excerpt is in IMPLICIT_TORQUE_SOURCE.txt. Normal contact sensor forces are measured; they are not full tangential force or the solver's internal contact active set. Exact hidden-drive-limit root cause must remain UNKNOWN without that evidence. No CPU velocity clipping, synthetic force, or altered hardware bounds was introduced.

## Remaining scope

No valid fullbody reference exists, so no new PhysX reference validation or128-case evaluation is authorized by these gates. Original Recovery/scoring/collision/actuator interfaces and protected physical files remain hash-identical. SuperMimic workspace, weights and jobs were not altered. Complete teacher closure is NOT_COMPLETED; the task-valid contact solver is not a native-success certificate.

### Completed calibration result

Historical PhysX replay completed800ticks and exited0 after approximately18minutes. All800 saved794D observations match the original trace exactly (max delta0), and final target command audit max delta0rad. The high-rate observer captured all2240 requested physics frames. No new planning or actor inference was executed.

CPU comparison (separate process exit0):

| Lane / prediction | Numerical solves | Failure / largest velocity error |
|---|---:|---|
|0, each step reset to native truth|14/14|max28.04537rad/s mismatch|
|0, cumulative from249|9/10|fails at258|
|1, each step reset to native truth|13/14|fails at250; max28.88643rad/s mismatch|
|1, cumulative from249|1/2|fails at250|

These are numerical solve counts, NOT physically feasible counts. At Lane0 interval259→260, FL_calf_joint actual velocity is−30.0000038rad/s while a one-step CPU prediction from the same native state gives−58.0453772rad/s. Thus mismatch exists even without accumulated state drift. The unchanged160-substep predictor cannot currently serve as a PhysX-equivalent certificate around joint velocity limits/contact transitions. Its exact missing reaction/solver mechanism remains UNKNOWN: the recorded torque is approximate and internal PhysX drive/contact impulses were not exposed. Calibration does not justify adding unbounded brake impulses or changing physical limits.

No new candidate reference was executed, and no new hit/Recovery/complete5 result is claimed. Calibration completion is distinct from teacher closure.
