# Strict event-coupled gate fix

The full-body rollout now accepts an explicit `impact_builder` callback tied to
the current candidate terminal trajectory. The callback must provide strict
finite-mass status and the four actual forward impact/task flags. Without it,
the rollout remains unverified. The planner now requires contact feasibility,
legal return, attack5 and target landing in addition to path, terminal
kinematics and finite-mass evidence.

This removes the previous fixed-label ambiguity and prevents a kinematic-only
candidate from passing as a complete CPU strike. Seven related tests passed,
exit 0. No native run was started because no candidate has supplied the
same-state impact builder evidence.
