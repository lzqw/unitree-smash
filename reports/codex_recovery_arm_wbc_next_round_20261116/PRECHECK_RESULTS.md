# Precheck Results

## CPU and controller checks

Command:

```bash
cd /home/autodrive/LZQW/unitree_smash_teacher_eval
bash scripts/python.sh -m pytest -q \
  tests/test_direct_sequence_search_contract.py \
  tests/test_sequence_repair_contract.py \
  tests/test_event_recovery_manager_v1.py \
  tests/test_native_recovery_sequence_replay_v1.py \
  tests/test_native_recovery_sequence_replay_v2.py
```

Result: **32 passed in 0.91s, exit code 0**.

## Engineering attempts

- Attempt 01: stopped before formal scoring because preview/native arbiter mismatch was detected; fixed by wiring the full-authority arbiter.
- Attempt 02: stopped before formal scoring because an adopted ndarray entered a list-style truth check; fixed by normalizing adopted sequence nodes.
- Attempt 03: same frozen candidate after those engineering fixes; completed 128/128 and 800/800 with exit code 0. Its raw logs and partial attempts are preserved under `attempts/`.

## Frozen protocol checks

- 128 environments and all first episodes present.
- 800 control ticks, 0.01 s control period, 8 s episode window.
- Seed `882072`, no noise, ground extent 64 m.
- `teacher_only=true`; learned action contributions all false.
- Numerical failures: 0.
- `command_audit.all_pass=true` in `TEACHER_DIAGNOSTICS.json`.
- `formal_eligible=false`, `formal_acceptance=false`, `formal_cases_read=false`.

## Not run / limitations

No second native candidate, ablation, parameter scan, RL/PPO/BC/distillation, or SuperMimic training was started. The saved contact telemetry is event-scoped; it gives 59 racket-contact point velocities, not a continuous joint-speed proof. The 58.48 rad/s value in the diagnostics is a command-target difference divided by 0.01 s, not a measured physical joint velocity.
