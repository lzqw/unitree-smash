# Push-off support QP replay

The existing causal push-off support QP was replayed with contact-window and
momentum-rate preservation. It adopted 11 QP commands across 29 intervals and
passed every path check. Terminal errors were position 584.672 mm, normal
0.258904, and racket velocity 4.392869 m/s. It is not a valid strike prefix.

The nonlinear push correction branch was also replayed. It adopted 10 QP
commands across 29 intervals and passed the path checks, but terminal errors
were position 596.859 mm, normal 0.269786, and racket velocity 4.422782 m/s.
It was rejected for teacher use.

The best retained CPU local refinement remains the leg-rate tolerant result
with 60.228 mm position error, 0.119973 normal error, and 1.856165 m/s velocity
error. Neither QP replay changed the default controller or Recovery path.
Native/PhysX evaluation, training and SuperMimic operation were not run.
