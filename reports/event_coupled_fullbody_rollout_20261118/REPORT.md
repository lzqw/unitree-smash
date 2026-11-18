# Event-coupled full-body rollout

Added `EventCoupledFullBodyRolloutV1`, exposing one 24-parameter candidate
vector containing twelve leg position corrections, six arm terminal position
corrections and six arm terminal-rate corrections. It delegates every
candidate to `CoupledContactRollout`, which generates actual 18D commands and
rolls the complete nonlinear state from the supplied initial state and command
history. No future state or contact mask is injected.

The rollout reports actual terminal errors and retains finite-mass status as
unverified until strict impact admission is connected to the same terminal
state. Six interface tests passed, exit 0. A production candidate sweep and
native PhysX validation are still pending.
