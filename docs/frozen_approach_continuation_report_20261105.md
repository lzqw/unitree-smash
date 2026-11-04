# Frozen approach continuation result

The 27/128 frozen approach policy was resumed for eight more PPO updates with 5,120 native environments, retaining its critic and optimizer state. The run completed 5,242,880 environment steps and 6,995 training episodes with zero numerical failures. The training stream produced **0 complete5** and **0 conditional recoveries** despite 3,585 airborne contacts; its peak value loss was 769.7.

The exported checkpoint was then evaluated natively on the fixed 128-case panel (seed 882072). It obtained **22/128 complete5**, 73 qualified airborne contacts, 34 conditional recoveries, 72 crossed-net cases, 63 target landings, and zero numerical failures. This is below the starting 27/128 policy, so the continuation is rejected.

The result confirms that simply extending the approach-head PPO run is unstable: the frozen checkpoint retained some behavior, but the training distribution stopped producing recoverable full events. The 27/128 policy remains the pure frozen baseline; the causal expert-bank controller remains a separate 28/128 controller trial. The next training change should add causal teacher selection or online correction memory instead of more updates from this drifted checkpoint.
