# Event-latch world-model GPU smoke

- Code commit: `38ebc22`
- Output: `/tmp/unitree_smash_g07_eventaware_smoke_20261114_093933`
- Device: `cuda:0`, Isaac GPU PhysX
- Environments: 2
- Iterations: 20
- Native transitions: 40
- World-model updates: 5
- Critic updates: 100
- Actor updates: 4
- Exit code: 0

The smoke created the event-aware replay layout, completed a 16-transition world window, ran imagined BPTT, and updated actor/critic without shape or import errors. No full physical episode was expected in 20 ticks, so this is an interface/runtime check rather than a task-performance result.

Final smoke diagnostics: world loss `0.3986`, imagined return `-5.1700`, native reward mean `-0.0115`, native event counts all zero. No formal evaluation was started.
