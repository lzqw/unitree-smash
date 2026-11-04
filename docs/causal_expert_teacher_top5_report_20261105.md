# Causal expert teacher with large-scale PPO

The new teacher-in-the-loop run used 5,120 native environments, the frozen v2 approach/recovery policy as initialization, 7 critic-only warmup updates, and a causal bank teacher blended at 0.35. To keep the native process tractable, the five historical experts with the largest case coverage were loaded. Each environment selected an expert from its first observed ball feature and held that selection until reset; uncovered cases fell back to the student action. The teacher was not embedded in the exported policy.

The training stream completed 6,945 episodes with **686 complete5**, 1,106 conditional recoveries, 3,541 airborne contacts, and zero numerical failures. After the eighth update, the student was evaluated with the teacher disabled on both fixed seeds:

| student evaluation | complete5 | air contacts | recovery | crossed net | target landings |
|---|---:|---:|---:|---:|---:|
| seed 882072 | 26/128 | 74 | 36 | 72 | 65 |
| seed 882075 | 26/128 | 74 | 36 | 72 | 65 |

The pure frozen baseline remains 27/128 and the explicit runtime expert-bank controller remains 28/128. This student is therefore retained as the next continuation initialization, not promoted as the best checkpoint. It is a substantially better transfer result than stateless action-MSE distillation (21/128): the causal teacher generated complete physical events in training, and the student preserved nearly all of the original capability after the teacher was removed.

The next continuation resumes this checkpoint with the same causal teacher and optimizer state so the approach head can learn from the complete-event distribution. The promotion gate remains a two-seed frozen evaluation with the teacher disabled.
