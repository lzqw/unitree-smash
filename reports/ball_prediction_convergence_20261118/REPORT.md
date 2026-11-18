# Causal ball prediction convergence

The locked causal ball state from observation tick 249 was propagated to the
three existing contact times with RK4 at 0.002 s and 0.000125 s steps. Position
differences were below 7e-14 m and velocity differences below 1.5e-14 m/s.
The current ball integration step is therefore not the source of the 58 mm
terminal racket-position error in the CPU candidate.

No future native state, native execution, training or formal evaluation was
used.
