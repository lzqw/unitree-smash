当前唯一GPU0作业（2026-10-24T17:14:00.762921+00:00核查）：ball_relative_prepose38_cem25b，guard1469660/child1469684，源码dba323d431defd826ef0afe344c6e4fdf7ee58d7，CEM25b使用相对公共来球预测引拍gain.35，参考实际CEM23 TRAIN65 env109；4代×64×TRAIN[1,65]，7200秒/409600控制步，首100步已验证。不要因观察EOF重启；不要手改GPU_LEDGER/RUNS，旧CEM24孤立guard已由reconcile_orphan_gpu_job.py封账。

CEM24四代已完成：512执行/409600步，310接触/83拍线/16攻击门槛/54拍线+恢复/6目标/0完整/NF0。第3代实际candidate11/TRAIN65同回合norm8.096750655、out[7.694765,-1.440564,-2.067061]、越网/场内首落、恢复3.021963s/body0，first[4.220016,-.746796]，目标超界.220016m；candidate48 norm8.175164、恢复2.723985s但firstx4.430500；所有完整标记0。权重/逐回合数据/最新视频已打包，包SHA70fb1277…（GitHub回执待核对）。

最新视频：https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_attack_recovery_20261025_1710.mp4 ，31.08秒/1554帧，真实CEM24candidate11两独立回合；字幕标注攻击恢复成立但目标落点未达标，另一来球失败，不能当完整成功。

CEM25首次启动只因未跟踪policy.pt缺失而失败（0.001397GPUh，未进入rollout），其job/error保留；复制并核验policy SHAacab8846…、bankSHAa0c6047b…后CEM25b已原生运行。relative_prepose_initialization_1715/REPORT.json证明运行时seed/参考体坐标/38维bank/动作流有限有界，0接触是尚未到来球阶段，不是失败结论。

仍保持原完整目标：真实完整球拍/标准悬空网/原规则，norm>=8、vx>0、vz<=-1、真实越网/原目标X[2,4],Y±1、合格跳跃/落足/连续.8s恢复、NF0；正式TRAIN128/扰动与A200+B200尚未运行。CEM25b完整后再决定是否扩大到独立面板。
