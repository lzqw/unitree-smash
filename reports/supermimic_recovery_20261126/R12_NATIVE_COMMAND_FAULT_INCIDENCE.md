## Native commands through the frozen recovery ensemble

On the R12 held32 lanes we compared risk scored on the recorded PhysX states with risk scored on each ensemble member's rollout. Each window starts from the same measured state and uses the exact recorded final joint targets. The same first-contact incidence model and calibration fit on R12 training lanes score both paths. Only windows active for their entire length contribute. The event tick is included.

| Window ending on first body contact | Measured-state median | Predicted-state medians across members |
|---|---:|---:|
| 8 ticks | 0.228 | 0.223, 0.239, 0.255 |
| 32 ticks | 0.346 | 0.225, 0.188, 0.298 |
| 48 ticks | 0.385 | 0.146, 0.160, 0.239 |

Each row uses the same 20 actual body-contact cases. Values are risk estimates on event-conditioned windows. At H48 the 13 windows ending on first relaxed recovery scored 0.071 on measured states and 0.052–0.100 on predicted states. Only two windows ended on first strict recovery.

Even with known native commands the H48 ensemble trajectories understate median danger on these fall windows. These measurements isolate a state rollout contribution and leave the sign of the action gradient unresolved. The ongoing R13 collection provides distinct launches for checking transfer. The audited ensemble is the original R12 model. The trained policy and deployed scoring remain unchanged.
