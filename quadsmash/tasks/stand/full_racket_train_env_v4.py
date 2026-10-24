"""TRAIN-only early termination after irrevocable rally/jump failure."""
from .full_racket_env_v4 import FullRacketEnvV4


class RallyTrainingEnvV4(FullRacketEnvV4):
    def __init__(self,*args,continue_after_miss=False,**kwargs):
        self.continue_after_miss=continue_after_miss
        super().__init__(*args,**kwargs)

    def _robot_rule_fault(self):
        return self.rally.robot_rule_faulted()

    def _get_dones(self):
        fallen,timeout=super()._get_dones()
        if not self._v4_ready:return fallen,timeout
        fault=self._robot_rule_fault() if self.continue_after_miss else self.rally.faulted()
        impossible=(self.launch_done&fault)|self.body_ground_seen
        if not self.continue_after_miss:
            impossible|=self.launch_done&self.rally.ground_seen&(~self.rally.landed_target)
        return fallen|impossible,timeout

    def _reset_idx(self,ids):
        reasons={};terminal_states={}
        if self._v4_ready:
            robot_rule_fault=self._robot_rule_fault()
            for i in ids[self._episode_has_stepped[ids]].cpu().tolist():
                reasons[i]=dict(rally_fault=bool(self.rally.faulted()[i]),
                    robot_rule_fault=bool(robot_rule_fault[i]),
                    shuttle_only_fault=bool(self.rally.faulted()[i] and not robot_rule_fault[i]),
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
            why=reasons[row['env_id']]
            row['training_early_termination']=(why['robot_rule_fault'] or why['body_ground'] or why['root_fallen'] or
                ((why['rally_fault'] or why['first_ground_without_target']) and not self.continue_after_miss))
            row['training_continue_after_miss']=self.continue_after_miss
            row['terminal_native_state']=terminal_states[row['env_id']]
