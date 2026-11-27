## Matched behavior experiment on the 256 R13 training launches

PPO01 and the absorbing-risk conservative candidate completed the same cases. On final_best lanes body contact was 71 → 71, strict recovery 11 → 6 and relaxed recovery 53 → 46. On h8d lanes the counts were 77 → 78, 7 → 7 and 46 → 52. These lane counts use the latched native event over all 128 launches per Approach. They do not have the qualified-contact denominator of fixed128.

121 cases had exactly matching recorded prefix, release tick and full release state. Of these, 13 were in the reserved WM holdout. For those cases we applied each branch's actual final joint commands to each frozen member from the common measured state. Aggregate angular velocity response cosines were 0.982–0.995 at H16, 0.706–0.720 at H32, and −0.111 to −0.016 at H48. Gravity direction response cosines were 0.990–0.991, 0.976–0.978 and −0.065 to +0.015 respectively. Each cosine pools vector components over cases. The H48 result shows little directional agreement on these releases. It does not establish a general sign reversal of the Actor gradient.

The next WM fit retains absolute dynamics supervision and adds differences between the measured trajectories at H16, H32 and H48. Only exact training-side release pairs contribute to that loss.
