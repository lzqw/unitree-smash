from quadsmash.rl.supermimic_physical_start_cache_v1 import APPROACH_TICKS,CRITIC_TICKS
from quadsmash.rl.supermimic_timebase_v2 import Timebase


def test_critic_windows_cover_entire_original_task_without_longer_rollouts():
    cfg=Timebase()
    covered=set()
    for tick in CRITIC_TICKS:
        covered.update(range(tick,min(tick+cfg.imagination_steps,800)))
    assert covered==set(range(800))
    assert cfg.imagination_steps==64 and cfg.world_steps==16
    assert CRITIC_TICKS[0]==768 and CRITIC_TICKS[-1]==0
    assert APPROACH_TICKS==(224,240,256,272)
