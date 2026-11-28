# Recovery gradient through the continuation value

On the same 24 native training starts as the [original gradient audit](R13_SEGMENT_GRADIENTS.md), we scaled the derivative of discounted survival times terminal value to 0.1. The numerical forward return, incumbent PPO01, actuator commands, G07 state, native Critic and three world models were held fixed. The default trainer still uses the original derivative. No policy weights were updated during this CPU audit.

| Starts | Original endpoint observation share | Scaled endpoint observation share | Scaled running reward share |
|---|---:|---:|---:|
| Impact or early prefault | 0.986–0.998 | 0.426–0.838 | 0.177–0.626 |
| Late prefall | 0.963–0.996 | 0.022–0.044 | 1.053–1.079 |
| Successful landed | 0.996–1.012 | 0.275–0.769 | 0.213–0.698 |

Each share is the signed projection onto the total gradient for that configuration. Shares may exceed one when another component points in the opposite direction. In the late-prefall group this adjustment gives the 16-step running reward most of the influence on the update. It has yet to demonstrate a reduction in native falls. The [raw results](r13_weak_tail_gradients.json) identify each sampled native start and show each member separately.

## Completed policy update and matched-action replay

Twenty optimizer steps took 800 seconds and sampled 298 late prefall starts. Comparing each new retained candidate against the corresponding previous candidate, with both action changes measured relative to PPO01 on identical native observations, gives a late-prefall cosine of 0.911–0.914 on the training side. Changing only the sampler previously left these directions at approximately 0.998. The directions changed, but they remain largely aligned. For the moderate candidate a single mean action-change vector within each late-prefall Approach group accounts for 89–91% of the change energy after equal weighting per case. The latter statistic measures similarity across states and does not establish that a common action is physically wrong. [Conservative replay](r13_weak_tail_action_conservative.json) · [Moderate replay](r13_weak_tail_action_moderate.json)

The native pulse also allows an oracle timing check on 22 training cases and six held cases with full return changes exceeding one unit. Using the true initial 16-step reward change plus the Critic gave the right direction on 13/22 and 3/6 respectively. Using 48 measured steps plus the same Critic gave 18/22 and 6/6. These longer trajectories use real PhysX observations and rewards after PPO01 resumed. The result provides no guarantee about 48-step WM predictions. An exploratory fit to paired native continuation values used 139 training endpoints and nearby post-intervention states. Across 80 optimizer steps it failed to improve the development direction count and paired RMSE together. The incumbent Critic was retained. [Native credit by horizon](r13_pulse_credit_horizons.json) · [Paired tail fit](r13_pulse_tail_fit.json)

Both resulting Recovery candidates have entered the deterministic 64-case simulation. No new fixed128 score was available at the time of this update.

The completed 64-case comparison retained the moderate candidate for fixed128 evaluation. In its own matched reference slot, body contacts went from 32 to 30 (7 prevented, 5 new), strict recoveries from 3 to 6 (4 gained, 1 lost), and relaxed recoveries from 8 to 20 (12 gained, none lost). Both versions had the same 54 qualified contacts, 46 ball-side successes and zero net faults, with no case flips in those metrics. In the conservative slot body contacts increased 31 to 33, strict recovery increased 3 to 4, and relaxed recovery increased 9 to 11. PhysX finished all 128 lane episodes across the two slots with zero numerical failures in 2,139 seconds. The separate official fixed128 simulation has started; PPO01 remains the confirmed checkpoint. [Paired outcomes by case](r13_weak_tail_native64_flips.json)

Among the 12 newly scored clean relaxed recoveries, seven avoided a previous body contact. Four of those seven had already reached the relaxed window before subsequently falling at baseline. The remaining five newly scored recoveries stayed clear of the ground in both runs and newly reached the stable window. The primary Approach gained seven clean relaxed cases and h8d gained five. [Case-level breakdown](r13_native64_gain_mechanisms.json)
