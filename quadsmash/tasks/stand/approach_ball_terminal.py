"""Lane-wise ball-only episode termination using measured native events."""
import math

import torch


def ball_terminal_mask(*, robot_terminal, had_contact, released, ground_seen,
                       physical_ground_seen, faulted, followthrough_pending,
                       time_since_release, followthrough_seconds, timeout):
    precontact_failure = ~had_contact & (robot_terminal | ground_seen | faulted)
    # A fault ends formal rally scoring. Keep the physical ball alive through
    # its first ground contact to preserve a measured failed-flight target.
    postcontact_fault = had_contact & faulted & physical_ground_seen
    flight_complete = had_contact & released & ground_seen & ~followthrough_pending & (
        time_since_release >= followthrough_seconds)
    return precontact_failure | postcontact_fault | flight_complete | timeout


class ApproachBallTermination:
    def __init__(self, count, device, protocol, dt):
        self.protocol = protocol
        self.followthrough_seconds = protocol['minimum_followthrough_ticks'] * dt
        self.tail_ticks = math.ceil(protocol['recovery_tail_seconds'] / dt)
        lanes = torch.arange(count, device=device)
        self.keep_tail = lanes % protocol['recovery_tail_lane_stride'] == 0
        self.physical_ground_seen = torch.zeros_like(self.keep_tail)
        self.robot_terminal = torch.zeros_like(self.keep_tail)
        self.ball_terminal = torch.zeros_like(self.keep_tail)
        self.numerical_abort = torch.zeros_like(self.keep_tail)
        self.end_tick = torch.full_like(lanes, -1)

    def observe_ground(self, *, launched, had_contact, touching_ground):
        self.physical_ground_seen |= launched & had_contact & touching_ground

    def step(self, rally, ticks, time, robot_terminal, numerical_abort, timeout):
        self.robot_terminal |= robot_terminal
        self.numerical_abort |= numerical_abort
        terminal = ball_terminal_mask(
            robot_terminal=self.robot_terminal, had_contact=rally.had_contact,
            released=rally.released, ground_seen=rally.ground_seen,
            physical_ground_seen=self.physical_ground_seen, faulted=rally.faulted(),
            followthrough_pending=rally.followthrough_crossed & ~rally.followthrough_finished & rally.in_play,
            time_since_release=time - rally.release_time, followthrough_seconds=self.followthrough_seconds,
            timeout=timeout)
        self.end_tick.copy_(torch.where(terminal & ~self.ball_terminal, ticks, self.end_tick))
        self.ball_terminal |= terminal
        retain = self.keep_tail & rally.had_contact & (ticks - self.end_tick < self.tail_ticks)
        return self.numerical_abort | (self.ball_terminal & ~retain & ~timeout)

    def record(self, lane, tick):
        return dict(robot_terminal=bool(self.robot_terminal[lane]), ball_terminal=bool(self.ball_terminal[lane]),
                    numerical_abort=bool(self.numerical_abort[lane]),
                    physical_ground_seen=bool(self.physical_ground_seen[lane]),
                    recovery_tail_observed=bool(self.ball_terminal[lane] and not self.numerical_abort[lane]
                                                and tick - self.end_tick[lane] >= self.tail_ticks))

    def reset(self, ids):
        for value in (self.physical_ground_seen, self.robot_terminal, self.ball_terminal, self.numerical_abort):
            value[ids] = False
        self.end_tick[ids] = -1
