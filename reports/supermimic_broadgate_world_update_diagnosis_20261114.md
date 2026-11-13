# Broad-gate run diagnosis

With a fallback threshold of 128 terminated episodes, the seventh run did not update the actor by iteration 331. It observed only eight early terminations, zero full timeouts, and six frame-hit events. The world loss rose to about `0.79`, so the `0.75` safety gate correctly remained closed.

This is a data/model coverage bottleneck rather than an actor update result. The next run will keep the broad gate and teacher-forced fit but use three world-model updates per native iteration so the model can track the changing native state distribution before any imagined policy update. Actor learning rate remains `1e-5`, anchor coefficient `0.2`, and all physical/task contracts remain unchanged.

