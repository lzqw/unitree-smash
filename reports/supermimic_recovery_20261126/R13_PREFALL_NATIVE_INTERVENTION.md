# R13 pre-fall intervention and Body WM refit

PPO01 remains the retained Recovery head. On official fixed128, the conservative localized candidate gave 6 strict and 20 relaxed recoveries with 74 body contacts among 108 qualified contacts. PPO01 had 10 strict, 25 relaxed and 71 body contacts. Ball-side completion was 95 versus 96 of 128. Neither policy incurred a robot-net fault.

## Native paired intervention

A separate experiment temporarily substituted the moderate candidate for PPO01 for 16 control ticks, then returned to PPO01. It completed 256 episodes, with 253 interventions and 4,048 changed commands. Of those episodes, 252 matched the entire native prefix exactly. Case B-931312-112 was excluded from paired scoring. Start times in the pre-fall group were selected retrospectively using the baseline fall time.

Among 145 matched episodes with a baseline fall, 58 fell later, 46 at the same tick and 41 earlier. Every one still fell. None of the other 107 paired episodes acquired a body contact. At H16 the median left rear thigh clearance response was approximately +5 mm across 127 training falls and +6 mm across 18 held falls. Median minimum whole-body clearance increased by less than 0.3 mm in both groups.

## Held action-response retest

After fitting on paired training windows, the ensemble passed the nominal crossing panel. The table gives H16 response cosine for each ensemble member on the 18 held pre-fall episodes. Each quantity compares the predicted change under the executed intervention with the observed native change.

| Quantity | Previous ensemble | Refitted ensemble |
|---|---|---|
| Left rear thigh clearance | .795, .896, .859 | .798, .893, .859 |
| Local trunk linear velocity | −.078, −.100, −.025 | −.084, −.104, −.029 |
| Local trunk angular velocity | −.128, −.080, .134 | −.175, −.180, .131 |

The refit provides no improvement in predicting the trunk response to this correction. It was scored offline and was not deployed. The next diagnostic should separate interventions that changed the native foot-support sequence from interventions with matching support, and compare the resulting trunk velocity changes.

## Foot support and the useful prediction horizon

Using the previous validated ensemble, we separated paired trajectories by whether any of the four observed contact flags differed at any control boundary within the window. In the 18 held pre-fall cases, H4 local trunk linear velocity response cosines ranged from .55 to .57 for 14 pairs with matching support sequences and were .70 for the four with changed support. At H16, the corresponding cosines ranged from −.41 to −.34 across seven matching pairs and approximately .00 to .11 across eleven changed pairs. Across all 18, the H4 angular response cosines were .65–.67, falling to −.13 to .13 at H16. These are aggregate response directions for one intervention. Matching binary contact flags do not establish equal contact forces.

A new first-block experiment uses four differentiable ticks from each measured G07 snapshot, retaining the 16-tick spacing between snapshot starts. It uses PPO01, the previous validated WM and native Critic. Twelve ticks between adjacent four-tick segments are outside the differentiable rollout. Native64 and official fixed128 retain their original full-tail protocols.

## Separating inherited trajectory error from four-tick velocity prediction

For each branch, the diagnostic restores its own measured pose and velocity four ticks before the endpoint, applies its recorded joint targets, and scores the difference between the two **changes** in world-frame trunk velocity. Subtracting each branch's starting velocity prevents a model from receiving credit for motion already present in the native state.

On the 18 held pre-fall pairs, averaging member cosines, the original model scored .599 linear and .663 angular on ticks 1–4. On ticks 13–16 the same diagnostic scored .510 and .289. In contrast, free-running from the original start to tick 16 scored −.068 and −.025 in local trunk coordinates. The frame conventions differ; the scores are not an additive attribution. H4 projected gain from the common prefix was only .46–.50 for world linear velocity and .58–.63 for angular velocity.

We fitted a separate CPU candidate for 36 updates per member on the existing training partitions, using the four-tick differential velocity loss alongside the original physical losses. Checkpoint selection used development pairs and kept the unmodified member as an option. The candidate passed the nominal crossing panel. On the held pairs, H4 world cosines moved to .601 and .664 and H16 open-loop local cosines to −.075 and −.089. This did not justify deploying the new weights. The active H4 Actor experiment uses the original validated ensemble.
