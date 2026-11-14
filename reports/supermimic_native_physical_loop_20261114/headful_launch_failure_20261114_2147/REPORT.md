# Headful launch failure

- Run: `calibrated_update_16_20261114_214647`
- Commit: `2d26e7f36fd7573d6e7811a5cbaf5e889271851a`
- Last lifecycle stage: `BEFORE_APP`
- No `AFTER_APP`, environment construction, reset, training, or checkpoint loading occurred.
- Kit log: `/home/autodrive/Isaac/env_isaaclab/lib/python3.11/site-packages/isaacsim/kit/logs/Kit/Isaac-Sim/5.1/kit_20261114_214702.log`

The crash occurred during RTX startup in `librtx.scenedb.plugin.so` while
headful Kit was launched with `/renderer/multiGpu/enabled=True`. This is a
renderer startup failure, not an actor, world-model, checkpoint, or data
contract failure. The orphan telemetry child was terminated after diagnosis.

The next command keeps headful Isaac, disables Kit multi-GPU rendering, and
uses the performance rendering preset. It does not modify PhysX or task
physics.
