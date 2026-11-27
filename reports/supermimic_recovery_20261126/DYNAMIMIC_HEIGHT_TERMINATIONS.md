# Recovery terminations at low root height

The native jump environment terminates an episode when the root height falls below 0.12 m. Its body-ground contact flag measures a separate event. The distinction affects interpretation of both the deterministic panel and the Critic targets.

## PPO01 fixed128

Among the 108 qualified contacts in the original full-tail PPO01 evaluation, first-episode terminal records give the following counts.

| Body-ground contact seen | Root-height termination | Timeout | Cases |
|---|---|---|---:|
| Yes | No | Yes | 70 |
| No | No | Yes | 26 |
| No | Yes | No | 11 |
| Yes | Yes | No | 1 |

Thus 11 of the 37 qualified cases without recorded body-ground contact ended through the root-height threshold. Counting these cases as trajectories that remained standing would overstate physical stability.

## First Dyna collection

In the 256-lane PPO01 archive, the intersection of eligible released frames, native terminal frames, root height below 0.12 m and neither body-ground nor net contact contains 37 independent episodes. All 37 had already landed. The lane split used for the Critic assigns 32 to training and five to validation. The recorded reward on the terminal frame ranges from +0.099 to +0.318. Termination removes subsequent rewards, but this transition does not receive the separate body-contact penalty. This observation alone does not establish the direction of the Actor gradient.

The counts use native terminal evidence from PPO01 and the sealed training archive. Candidate effects require a paired comparison, including whether fewer reported body contacts coincide with more low-height terminations.

## Tail-value and contact-risk checks

The Critic fits frames up to the first observed body contact. For episodes that continued in PhysX after contact, we scored the very next observation with the same frozen Critic used for the first BPTT block. The model returned a positive value on 16 of 153 training lanes and two of 20 held lanes. Its median values were −12.56 and −11.81 respectively. Recovery returns are truncated at contact in the native targets. Unconditionally adding the model's tail value after a predicted failure can therefore introduce a spurious reward or penalty in imagination.

To check whether a predicted contact can safely substitute for the native flag, we ran one WM step from measured states with the recorded command at two times in the same 20 held episodes. At first native body contact, the three-member mean probability exceeded 0.5 in 18/20 cases (median 0.568). Twelve control ticks earlier it already exceeded 0.5 in 10/20 (median 0.499). Only 14/20 probabilities increased between those states. These are event-aligned counts, not estimates of the false-positive rate across all native frames. A hard cut at 0.5 would remove valid pre-contact frames on this panel.
