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
        reasons={}
        if self._v4_ready:
            for i in ids[self._episode_has_stepped[ids]].cpu().tolist():
                reasons[i]=dict(rally_fault=bool(self.rally.faulted()[i]),
                    first_ground_without_target=bool(self.rally.ground_seen[i] and not self.rally.landed_target[i]),
                    body_ground=bool(self.body_ground_seen[i]),
                    root_fallen=bool(self.robot.data.root_pos_w[i,2]<.12))
        start=len(self.rally_records)
        super()._reset_idx(ids)
        for row in self.rally_records[start:]:
            row['training_terminal_reasons']=reasons[row['env_id']]
            row['training_early_termination']=any(reasons[row['env_id']].values())
