# Event-conditioned whole-body smash baseline

The dependency-light backend now exposes the same contract intended for Isaac
Lab:

- `QuadSmashEnv` emits a single 18-D action (`12` legs + `6` arm joints).
- Observations include remaining hit time, racket pose/velocity, phase, contact
  flags, and the previous action. The critic can consume the complete state.
- `CurriculumStage` progresses from jump/recovery to virtual events, real
  contact, smash-and-recover, and domain randomization.
- Racket contact uses swept geometry and restitution. Contact is an event pulse;
  the episode continues until the shuttle outcome and stable recovery are
  known.
- `ReachableStateBank` stores complete finite dynamic states for reset-only
  intermediate-state initialization. It never mixes old trajectories into PPO.
- `RecoveryFeasibilityPredictor` and `select_event` implement the optional
  recovery-aware contact-event selector with a conservative uncertainty bound.

## Smoke training

```bash
PYTHONPATH=. python scripts/train_ppo.py \
  --steps 128 --num-envs 32 \
  --curriculum-stage virtual_event \
  --checkpoint checkpoints/whole_body.npz
python scripts/evaluate_policy.py --checkpoint checkpoints/whole_body.npz
```

The NumPy PPO is a reproducible API baseline. Isaac Lab can replace the
reduced-order dynamics while retaining the goal, event, reward, and reset-bank
interfaces.
