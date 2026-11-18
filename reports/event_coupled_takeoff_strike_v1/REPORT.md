# Event-coupled takeoff and strike planner P2 interface

Added a bounded outer planner that accepts takeoff, hit, push, base, arm and
string parameters through a fixed-dimensional vector and delegates every
candidate to a caller-owned full 18D rollout. The planner never writes future
q/v, contacts or forces. It gates position <=3 mm, normal <=0.12, velocity
<=0.75 m/s, path validity and `FINITE_MASS_VERIFIED`. Analytic or unverified
mass results fail closed. Native teacher eligibility remains false.

This is the first executable P2 interface. A production rollout callback still
must be connected to the existing WBC/fullbody predictor before candidates can
be evaluated. Five CPU tests passed, exit 0. No native run or SuperMimic
operation occurred.
