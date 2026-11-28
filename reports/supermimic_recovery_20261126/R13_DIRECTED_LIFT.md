# Recovery request saturation and landing lift

The weak-tail run finished its fixed128 evaluation with 77 body contacts, 5 strict and 22 relaxed recoveries among 108 qualified contacts, 96 ball-side returns and zero robot-net faults. PPO01 remains the reference. A later H4 candidate is in native validation.

## Request and backward checks

We reconstructed the 18 raw actuator requests ten control ticks before first body contact on all 146 eligible baseline falls. The replay of the clipped request agreed with the recorded command within 1.8e-7. The left rear calf raw mean was −1.189, with base contribution −0.00032 and Recovery contribution −1.189. This joint clipped on 139/146 cases, and on 133/146 when replaying the weak-tail moderate candidate. The other frequently saturated legs were front right thigh (135), left rear thigh (74), right rear calf (74), front left hip (67), and front right calf (62). [Per-joint evidence](r13_fault10_requests.json)

The new BPTT option preserves the native clipped forward request. It passes upstream descent directions that point back into the box and suppresses outward directions beyond the boundary. This is a surrogate derivative, and a parameter update must still clear the boundary to change a saturated native command. During a four-tick CPU replay through G07, the physical world model, task reward and tail, each ensemble member supplied an inward gradient at 14/18 saturated left-rear-calf positions in the early group. The late group supplied an inward direction at 9, 9 and 12 of 28 positions. The diagnostic uses eight cases per group. These counts are lane-ticks and cannot establish native recovery improvement. [Backward evidence](r13_directed_clamp_gradients.json)

## New training block

The new option applies an extra landed penalty to leg requests outside ±0.95, emphasizing the left rear thigh and calf. Height reward rises linearly over 0.12–0.30 m, and leg-speed damping ramps up between 0.15 and 0.20 m. The airborne objective encourages smaller rear-to-front foot height difference and retains the existing attitude and angular velocity terms. Sampling starts at least 50 ms after release for airborne cases and reserves distinct cases in the first 24 ticks after landing, alongside late pre-fall starts. The 21-surface geometry, 75 mm rear-thigh target and 0.1 tail-gradient scale stay active.

The trainer supports 24/32-step differentiable segments. This first comparison uses four ticks from native anchors because the independently measured pre-fall action response loses directional accuracy at H16/H32. Longer continuous airborne segments still need a phase-specific action-response check. The next block is queued behind the current H4 official evaluation and selects its retained checkpoint. Neither a split head nor a new Approach distribution is part of this first change.
