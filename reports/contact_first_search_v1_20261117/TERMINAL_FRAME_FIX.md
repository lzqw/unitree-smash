# Terminal evaluator frame correction

The planner solved IK in the predicted future body frame. It previously sent
that frame's selected normal and velocity to the physical evaluator alongside
a position in the fixed current planning frame. They could not be consistently
compared against a measured racket state after base rotation.

All three terminal targets now use the same fixed current planning frame.
Future-body-frame quantities remain available under explicitly named audit
keys. The physical measurement callback contract now explicitly requires this
same fixed frame. Native world measurements need an explicit transform when
the caller plans in initial body coordinates.

Also corrected the two-element causal target callback's default normal: use
the original current-frame normal, not the mutable future-frame IK variable.

Tests: 41 passed, exit 0, including the tuple-default and rotated evaluator regressions.
Actual output: TERMINAL_FRAME_TEST_OUTPUT.txt.
This corrects prediction admission metadata; it does not establish outgoing
shuttle quality or a native improvement. Outgoing-state integration remains
pending. No native evaluation or SuperMimic task was started.
