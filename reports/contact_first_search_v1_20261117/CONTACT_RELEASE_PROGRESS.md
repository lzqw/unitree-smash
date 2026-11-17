# Optional unilateral release in joint prefix prediction

The full-body evaluator can select `contact_prediction="unilateral_release"`.
It reuses the existing active-set implicit PD model, including full robot
inertia, unilateral contact and Coulomb friction. Its default remains
`measured_fixed`; no Recovery controller configuration was changed.

Only initially measured feet are candidates. Release follows the solved
contact dynamics; no force or future support schedule is injected. Released
feet are not reintroduced in later intervals. This is a pre-contact takeoff
model component, not a touchdown/landing predictor. Force traces and actual
contact changes are recorded. Motor-torque diagnostics are now explicitly
returned rather than inferred from a missing diagnostic value. The predictor's
existing equations and active-set search were not changed.

Actual CPU findings:
- One measured foot with an upward-separating full-body state releases; all
  subsequent foot forces are zero. The copied initial state is unchanged.
- Four measured feet in the same synthetic high-upward-velocity state failed
  active-set convergence. The initial test exited 1 (1 failed, 16 passed).
  This failure is retained as a rejection regression, not converted into a
  successful takeoff claim or a bilateral fallback.
- Final focused suite: 37 passed, exit 0 (CONTACT_RELEASE_TEST_OUTPUT.txt).
  Passing includes the expected rejection; it does not mean every physical
  query is solvable.

Native confirmation: NOT_RUN. Support-driven takeoff optimization and its
execution still remain incomplete. The action-source boundary question also
remains open: the historical base contains learned actors, while the requested
pure-teacher contract forbids their contribution. No native evaluation or
SuperMimic training was started.
