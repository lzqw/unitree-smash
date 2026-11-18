# Event-Coupled V4 optimizer correction

`optimize()` now has an explicit bounded rollout budget and deterministic
coordinate trust probes. Every center and probe receives the same formal gate.
Probe acceptance is no longer hard-coded. Results are cached by exact theta
bytes. The optimizer separately preserves `best_feasible`, `best_contact` and
`best_kinematic`, along with the full trace and failure reasons.

This corrects the prior optimizer bookkeeping. It does not claim a feasible
strike because the production candidate sweep still requires the long-running
full-body rollout harness and same-candidate impact builder. Six related tests
passed, exit 0. Native PhysX was not run.
