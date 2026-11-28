# Recovery gradient through the continuation value

On the same 24 native training starts as the [original gradient audit](R13_SEGMENT_GRADIENTS.md), we scaled the derivative of discounted survival times terminal value to 0.1. The numerical forward return, incumbent PPO01, actuator commands, G07 state, native Critic and three world models were held fixed. The default trainer still uses the original derivative. No policy weights were updated during this CPU audit.

| Starts | Original endpoint observation share | Scaled endpoint observation share | Scaled running reward share |
|---|---:|---:|---:|
| Impact or early prefault | 0.986–0.998 | 0.426–0.838 | 0.177–0.626 |
| Late prefall | 0.963–0.996 | 0.022–0.044 | 1.053–1.079 |
| Successful landed | 0.996–1.012 | 0.275–0.769 | 0.213–0.698 |

Each share is the signed projection onto the total gradient for that configuration. Shares may exceed one when another component points in the opposite direction. In the late-prefall group this adjustment gives the 16-step running reward most of the influence on the update. It has yet to demonstrate a reduction in native falls. The [raw results](r13_weak_tail_gradients.json) identify each sampled native start and show each member separately.
