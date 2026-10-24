"""TRAIN-only early termination after irrevocable rally/jump failure."""
from .full_racket_env_v4 import FullRacketEnvV4


class RallyTrainingEnvV4(FullRacketEnvV4):
    def _get_dones(self):
        fallen,timeout=super()._get_dones()
        if not self._v4_ready:return fallen,timeout
        impossible=(self.launch_done&(self.rally.faulted()|
            (self.rally.ground_seen&(~self.rally.landed_target))))|self.body_ground_seen
        return fallen|impossible,timeout

    def _reset_idx(self,ids):
        reasons={};terminal_states={}
        if self._v4_ready:
            for i in ids[self._episode_has_stepped[ids]].cpu().tolist():
                reasons[i]=dict(rally_fault=bool(self.rally.faulted()[i]),
                    first_ground_without_target=bool(self.rally.ground_seen[i] and not self.rally.landed_target[i]),
                    body_ground=bool(self.body_ground_seen[i]),
                    root_fallen=bool(self.robot.data.root_pos_w[i,2]<.12))
                data=self.robot.data
                terminal_states[i]=dict(
                    scope='Actual native state at terminal control step, before automatic reset; not exact first-collision substep',
                    time_s=float(self.episode_length_buf[i]*self.step_dt),
                    body_link_positions_m=(data.body_link_pos_w[i]-self.scene.env_origins[i]).cpu().tolist(),
                    body_link_quaternions_wxyz=data.body_link_quat_w[i].cpu().tolist(),
                    ball_position_m=(self.shuttle.data.root_pos_w[i]-self.scene.env_origins[i]).cpu().tolist(),
                    ball_velocity_mps=self.shuttle.data.root_lin_vel_w[i].cpu().tolist(),
                    root_com_velocity_mps=data.root_com_lin_vel_w[i].cpu().tolist(),
                    root_angular_velocity_rad_s=data.root_com_ang_vel_w[i].cpu().tolist())
        start=len(self.rally_records)
        super()._reset_idx(ids)
        for row in self.rally_records[start:]:
            row['training_terminal_reasons']=reasons[row['env_id']]
            row['training_early_termination']=any(reasons[row['env_id']].values())
            row['terminal_native_state']=terminal_states[row['env_id']]
