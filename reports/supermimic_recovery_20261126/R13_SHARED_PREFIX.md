# Recovery after a shared 290-tick native prefix

All 128 lanes used PPO01 for the first 290 native control ticks. The run then switched to the conservative segmented Recovery candidate for the rest of each episode. This diagnostic was excluded from promotion. Both policies share the frozen Approach, base and normalization. The last observed release under PPO01 was at 2.8582 s.

| Fixed128 metric | PPO01 | Candidate from the start | Candidate after tick 290 |
|---|---:|---:|---:|
| Qualified string contact | 108 | 108 | 108 |
| Body contact among qualified cases | 71 | 77 | 79 |
| Strict recovery among qualified cases | 10 | 8 | 6 |
| Relaxed recovery among qualified cases | 25 | 17 | 16 |
| Ball-side success | 96 | 94 | 96 |
| Robot net faults | 0 | 0 | 0 |
| Strict Complete5 | 10 | 6 | 5 |

The stored outgoing velocity, first landing, qualified contact, ball-side score and robot net flag all matched PPO01 case by case under the shared prefix. The body contact flag changed in 17 of the 128 cases. The later actions of this candidate yielded worse recovery on the panel even with identical measured ball outcomes. The comparison does not identify which joint action produced a fall. The altered switch schedule is a diagnostic intervention, and the retained checkpoint is PPO01.

The separate late-fall sampler experiment starts again at PPO01. Its training does not consume any fixed128 episodes. The candidate in this report predates that sampler.
