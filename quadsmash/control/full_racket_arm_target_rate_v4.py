"""Optional software slew bound on arm position targets; no physical state writes."""
import math
import torch


class ArmTargetRateV4:
    def __init__(self, count, device, maximum_rate_rad_s, control_dt_s=.01):
        # The historical teacher used 5 rad/s.  Residual learners may opt into
        # a larger *software command* slew rate; the physical joint limits and
        # native actuator still remain active downstream.
        if not math.isfinite(maximum_rate_rad_s) or not 0 < maximum_rate_rad_s <= 20.:
            raise ValueError('Arm target rate must be positive and at most20rad/s')
        if not math.isfinite(control_dt_s) or control_dt_s <= 0:
            raise ValueError('Require a positive finite control period')
        self.rate = float(maximum_rate_rad_s)
        self.control_dt = float(control_dt_s)
        self.target = torch.zeros((count, 6), device=device)
        self.clock = torch.zeros(count, device=device)
        self.initialized = torch.zeros(count, dtype=torch.bool, device=device)

    def reset(self, ids=None):
        if ids is None:self.initialized.zero_()
        else:self.initialized[ids]=False

    @torch.inference_mode()
    def __call__(self, desired, observed, time):
        if desired.shape != self.target.shape or observed.shape != desired.shape or time.shape != self.clock.shape:
            raise ValueError('Require six arm targets/observed joints and one public clock per lane')
        if not all(torch.isfinite(x).all() for x in [desired, observed, time]):
            raise ValueError('Finite public inputs required for arm target rate control')
        restarted = (~self.initialized) | (time < self.clock-1e-6)
        before = torch.where(restarted[:, None], observed, self.target)
        dt = torch.where(restarted, torch.full_like(time, self.control_dt),
                         (time-self.clock).clamp(0., self.control_dt))
        maximum_step = self.rate*dt[:, None]
        self.target.copy_(before+(desired-before).clamp(-maximum_step, maximum_step))
        self.clock.copy_(time)
        self.initialized.fill_(True)
        return self.target.clone()
