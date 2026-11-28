# Recovery gradient at the incumbent policy

Three groups of eight native R13 starts each contained four lanes per Approach. For each group, all three WM members used the same starts, recorded G07 snapshots, H16 horizon, and PPO01 Critic. The first imagined actuator target was checked against its native recording. No optimizer step was taken. The JSON contains the tick and lane for each start and each component's parameter-gradient norm. The signed projected share is the dot product with the total gradient divided by the squared norm of the total gradient. Opposing components can cancel.

| Native starts | Critic through endpoint observation, projected share | Discounted running reward, projected share | Critic through survival, projected share |
|---|---:|---:|---:|
| Impact or early prefault | 0.986 to 0.998 | −0.002 to 0.057 | −0.043 to 0.012 |
| Late prefall | 0.963 to 0.996 | 0.741 to 0.829 | −0.792 to −0.737 |
| Clean successful landing | 0.996 to 1.012 | −0.018 to 0.002 | −0.001 to 0.005 |

At the late prefall starts, the endpoint value's proprioceptive observation accounted for 0.664–0.700 of the total projected direction, and its racket features for 0.248–0.273. The immediately preceding action accounted for 0.030–0.036. At these starts much of the running-reward gradient cancels the direction obtained by differentiating survival times the tail value. The final direction of the unmodified training objective tracks the Critic's sensitivity to the predicted endpoint. A projected share does not validate that direction against PhysX.

The late-prefall sampler completed 20 BPTT updates in 769 s, including 298 measured late-prefall starts. On recorded late-prefall observations the new conservative candidate and the previous conservative candidate have action-difference cosines of 0.998 for both Approach policies, relative to the same PPO01 actions. This replay establishes that larger coverage left the overall action direction nearly unchanged. Native evaluation remains decisive.
