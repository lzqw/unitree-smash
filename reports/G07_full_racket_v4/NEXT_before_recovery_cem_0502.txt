# 当前唯一GPU：ball_push_grid01，继续解决全身旋转与恢复

目标不缩减，继续前读 /home/autodrive/.codex/attachments/55e175f0-9444-4a09-ad37-681e6586ca51/pasted-text-1.txt。仅GPU0，无累计GPUh/时间窗/截止，有限每job保护；仍须实际完整扣杀、A/B各200原门槛及完整交付，当前没有完整成功。

活跃exec65283，guard1332513/child1332525，18候选×TRAIN[1,65]×1代，最多8s/62.5us物理/10ms控制。2400s/28800控制-env步cap。先查同一PID/progress/error/job.json或poll同一exec；观察超时不是终止，不重复启动，勿编辑guard持有的GPU_LEDGER/RUNS。

新证据：ball_timing_grid01原生接触中，机身根转动对触点世界X贡献约-5.37..-5.56m/s，机械臂相对贡献+2.0..+2.99、根平移+.21..+.37，所以实际触点仍向己方。实际根pitch-24..-48deg、omegaY-4.49..-5.32rad/s。CONTACT_VELOCITY_DECOMPOSITION.json保留精确分解，不能将分解当干预成功。

当前保持相同触球arm/时序与物理，仅扫描16种前后腿push_thigh/calf偏置各[-.15,-.05,.05,.15]，另2参考候选，原动作/执行器限幅不变。代码新增--push-grid及采样契约；warm-start实际重测，不继承旧fitness。新TRAIN日志在全候选接触窗口也保存794D公开观测，终止前保存实际全身/球状态（控制步时刻，不伪称精确首次身体接触子步）。当前源已随guard快照。

上批ball_timing_grid01完成32首回合含重复参考：3真实合法空中拍线、1拍框、0完整/NF0；三次拍线都身体触地后终止。实际391.810751s/.108836320GPUh/10176步，保守账本cap25600。最新28.94秒/1447帧视频已完整解码无错误并上传SHA核验：
https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_air_contact_20261024_0450.mp4
含实际侧视俯视慢放/接触子步快照/拍框首落点；0完整，不能称独立400通过。最新代码已push a1b495c，当前pushgrid新更改仍待提交。

后续按当前实际结果继续联合优化（必要时姿态/角速度反馈、接触法向速度跟踪、不同有效种子或残差学习），扩大TRAIN128与原扰动，原生近网整拍/随挥边界与外部USD来源锁，冻结源码权重，独立原A/B200。目标仍实际己方合法空中拍线+攻击出球>=8m/s/vx>0/vz<=-1+真实越网首次目标落点+同回合无bodyground稳定.8s；worstpanel完整>=.5/空中>=.7/条件恢复>=.85/NF0。不能用几次弱触球/CPU测试/静态IK说完成或不可行。r3资产/675mm整拍/153g拍夹具/20.709991kg总重/真实悬空网和原协议不变，历史失败完整保留。不改preexisting status/G07_RESOURCE_STOP.json，不停止无关MPS/GPU任务。

更早上下文见reports/G07_full_racket_v4/NEXT_before_push_grid_0454.txt及此前NEXT快照。
