# Event-coupled rollout adapter

Added an adapter that converts a complete existing 18D rollout result into the
strict event-planner evidence contract. It reads the actual terminal errors and
trajectory from the rollout, propagates finite-mass status explicitly, and
fails closed on an invalid path. It does not synthesize q/v, contact state or
impact output. Teacher eligibility remains false until native validation.

Six related tests passed, exit 0. Native evaluation and SuperMimic were not run.
