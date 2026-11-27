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
