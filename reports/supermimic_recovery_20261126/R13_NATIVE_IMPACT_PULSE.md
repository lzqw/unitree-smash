# Sixteen native Recovery steps near touchdown

All 256 PhysX episodes finished. At tick 336 both rollouts were eligible in 252 cases and had identical prior observations, states, actions, and actuator targets through the intervention input. The six Recovery tensors changed for ticks 336–351 on both Approach versions. All lanes used PPO01 again from tick 352 onward. The [switch replay](r13_pulse_switch_execution.json) reproduced the stored action on both sides of each boundary to within 0.0000003.

| Paired complete episodes | PPO01 | Pulse |
|---|---:|---:|
| Body touched ground | 145 | 146 |
| Low-root termination while still eligible without a body or net fault | 45 | 41 |
| Clean strict recovery | 17 | 18 |
| Clean relaxed recovery | 53 | 50 |
| Robot net fault | 0 | 0 |

The apparent one-case reduction in body contacts at tick 416 did not persist. At termination no baseline fall was prevented and one new body contact occurred. Four fewer low-root terminations show why the separate failure counts matter. These are training-archive cases with a mixed Approach distribution and a common continuation. No fixed128 promotion decision is based on them.

On training cases still eligible at both endpoints, the measured prefix reward was removed from both the full native return and H16 estimate. For 20 cases with a material continuation change, the Critic predicted its direction correctly 12 times. Over all 162 eligible endpoints its RMS difference was 2.00, compared with 7.01 in PhysX. The held partition had five material changes with two correct directions. The counts are too small to establish a generally reversed gradient. [Paired case data](r13_pulse_continuation_pairs.json)

The native impact data also supported 186, 182 and 146 independent training cases at H4, H8 and H16 respectively. After fitting 18 updates per WM member and selecting by development response and H32 accuracy, all three members still passed the absolute H32 gates. On the 24 held H16 pairs, the weakest member's root linear velocity direction cosine changed from −0.129 to −0.088, and angular velocity from 0.176 to 0.219. The measured responses provide limited support for replacing the incumbent ensemble. [Before](r13_pulse_wm_response.json) · [After](r13_pulse_fitted_response.json)

The predeclared training-side continuation check triggered the 20-update block with 0.1 continuation gradient scale. The incumbent WM, PPO01 starting policy and native Critic were retained to isolate the change in gradient. The first GPU update completed. Policy selection still requires native evaluation.
