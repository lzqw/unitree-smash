# Change Plan and Evidence

| Area | Evidence before change | Change | Verification |
|---|---|---|---|
| Recovery history | Direct search raised `Actual command history required`; isolated interface refused inferred history | Bind the real 18D previous final command on every prediction/bounds/search path | Focused CPU tests; native 28,687 command rows with history mismatch max `3.62e-7 rad` |
| Preview/dispatch | Prediction used old arbiter while native Recovery used a different transform | Use `FullAuthorityCommandArbiterV1` in the native teacher and full-authority preview | Native preview-vs-dispatch max `0`; dispatch round-trip `1.79e-7 rad`; zero `>2e-6` rows |
| Arm target slew | Analytic teacher arm target rate and Recovery software slew added an extra limit | Disable only those software limits in teacher candidate | `teacher_arm_target_slew_removed=true`; native actuator/position/velocity/effort/collision limits unchanged |
| Direct search cache | Adopted `ndarray` entered list-style truth/length cache | Normalize each adopted node to an independent Python list/array | Full native run completed; two 32-node sequences adopted |
| Learned action path | Teacher-only requirement | Keep all learned action contributions disabled | Actor forward count 0; optimizer step count 0 |

No physical model, asset, collision, scoring, episode protocol, SuperMimic workspace, or training weight was changed.
