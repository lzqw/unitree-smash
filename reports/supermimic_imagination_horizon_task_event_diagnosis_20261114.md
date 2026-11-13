# Imagination horizon diagnosis

The stable native stream now contains real events, but the critic-quality gate did not open the actor: after more than 600 critic updates, critic loss remained hundreds and imagined returns were negative. The current imagination horizon is `0.96 s` (96 control steps), while the configured shuttle launch/impact event occurs around `2.7 s` after reset. Therefore the imagined objective cannot directly observe the primary legal string-contact reward.

The next run will use a `3.2 s` imagination horizon (320 control steps) and a smaller imagination batch (`1024`) to fit GPU memory. The world-model fit remains 0.32 s teacher-forced windows. This is the first horizon change in this tuning sequence; the task physics, event definitions, action contract, and native collection remain unchanged.

