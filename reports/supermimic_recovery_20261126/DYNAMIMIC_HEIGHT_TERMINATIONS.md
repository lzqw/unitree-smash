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

## Native candidate outcome and next block

For the moderate proposal, five qualified cases changed from body contact to no body contact. Two achieved strict recovery, while three ended because root height fell below 0.12 m. For the conservative proposal both such cases ended at the root-height cutoff. This split motivates counting native root failure when training the next candidate.

The revised training target gives the same one-off fall cost to an eligible native low-root termination without a body or net fault. The revised imagination objective attenuates future rewards and terminal value through a smooth survival weight over the 0.11–0.13 m height interval. It latches the worst survival reached and charges only the decrease, so a rebound cannot restart a failed episode. Native success definitions and promotion gates are preserved. Pre-failure sampling also covers these low-root terminations.

A fresh Critic trained on the same 37,924 native training frames. Its 6,169 held frames give explained variance 0.603 and MAE 9.85 in the revised return units. The fresh Critic and accepted WM passed a CPU H48 backward check from two recorded starts, including reproduction of the first executed joint targets. Policy gains from these revisions still require native evaluation.

On the five held native cases that terminated below 0.12 m without a contact flag, the three WM members predicted 5, 4 and 5 crossings when initialized eight ticks before the event and given recorded commands. At 32 and 48 ticks all three missed all five crossings. Mean endpoint height bias was +1.64 cm at 32 ticks and +2.48 cm at 48 ticks. These event-conditioned results are separate from the 32-case touchdown accuracy panel and measure sensitivity only.

Training has 32 independent eligible snapshots within 16 ticks of a low-root-only termination (17 primary, 15 companion). The revised sampler reserves about one quarter of the touchdown allocation for these snapshots whenever available. It keeps the 30/40/30 phase mix, equal Approach counts and full H48 rollout, and logs the actual count per update. The fixed action-budget check draws from the original phase allocation. The five held failures remain excluded from training.
