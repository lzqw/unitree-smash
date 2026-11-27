## R13 native collection and recovery world model

All 256 launches completed by tick 800. The original frozen PPO01 Recovery head generated both panels. The case mix changed between R12 and R13. Counts across panels describe available training data.

| Archive and Approach | Non-held lanes | Active body fault | Clean strict recovery | Clean relaxed recovery |
|---|---:|---:|---:|---:|
| R12 final_best | 112 | 79 | 3 | 14 |
| R13 final_best | 112 | 61 | 9 | 25 |
| R12 h8d | 112 | 74 | 9 | 21 |
| R13 h8d | 112 | 66 | 6 | 22 |

Clean recovery excludes any live body contact, net fault or native low-root termination. On R13 held32, the frozen R12 incidence model expected 15.12 first body contacts against 19 observed. Median H48 risk on measured states was 0.393 at first body contact, 0.119 at first relaxed recovery and 0.038 at first strict recovery (19, 13 and 2 windows).

The ensemble trained on the two archives passed its R13 H32 check. Member tilt error medians ranged from 2.04 to 2.51 degrees and center-of-mass height MAE from 4.02 to 5.26 mm. Recovery BPTT started after validation. Policy promotion requires separate native evaluation.
