# Causal expert teacher blend 0.2

Starting from the 27/128 frozen v2 policy, the top-five causal expert teacher was blended at 0.2 during an eight-update, 5,120-environment PPO run. The teacher selected an expert from the first observed ball feature and held that selection for the episode; it was removed from the exported student before evaluation.

Training produced 6,988 episodes, 619 complete5 events, 1,073 conditional recoveries, 3,536 airborne contacts, and zero numerical failures. The exported student scored **26/128 complete5** on both fixed seeds, with 74 air contacts, 36 recoveries, 74 crossed-net cases, 65 target landings, and zero numerical failures.

The result matches the blend-0.35 student (26/128) and remains one case below the pure 27/128 baseline. It confirms that the causal teacher transfers useful behavior without the severe 21/128 degradation of stateless action-MSE distillation, but lowering the blend alone does not improve the final student. The next experiment should restrict teacher influence to states with measured recovery benefit or train a dedicated recovery module while preserving the current approach mapping.
