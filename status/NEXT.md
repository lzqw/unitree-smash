当前唯一GPU0作业：residual_ppo_pilot_train128_1230，guard1559746 / child1559760，session35701。源码提交5b978ac。先查真实进程、INPUTS/progress、error和completed/job；禁止重复启动。

继续前重读目标：/home/autodrive/.codex/attachments/55e175f0-9444-4a09-ad37-681e6586ca51/pasted-text-1.txt。只用现有GPU0，无累计GPU/工作窗上限；保留每job故障保护和真实资源计费。原物理/规则/验收门槛不变。GPU0实测RTX5080 16GB，无云资源。

当前已启动stock RSL-RL原生残差PPO：全部128个固定TRAIN案例、128并行环境、8个更新，每更新128控制步，总131072控制环境步；62.5us物理/10ms控制/8秒回合。固定基线为CEM32横向candidate12，SHA51a8b19b17e19e60bd7482c7bcb7ed1d4e8b7f50b9317cac3c49007089e8b62d。源仍为UV[0,.10]、航向增益0、无高度自适应；本次不继承任何基线成绩。网络均值初始0，log_std0.1；前两次更新冻结actor和log_std，critic热身；之后学习18维全身残差。腿/臂目标增量上限.12/.15rad，经过原关节范围与5rad/s臂目标速率限制，再进入原生执行器；力矩/速度/资产没有改。

PPO脚本scripts/train_full_racket_residual_ppo_v4.py；配置configs/g07_full_racket_residual_ppo_v4.json。训练奖励只来自真实事件和当前因果预测的接近程度；完整链奖励只在实际回合终点发放，原complete标志不变。恢复截断使用既有FinalObservationPPO的真实reset前观测。每lane独立重置规划/行走/恢复/限速缓存，不重置其他lane；行走actor保留世界50Hz节拍。变动权重的训练回合计数不能当作固定策略成绩。

322检查通过；residual_ppo_ready_1220/REPORT.json证明800步、2lane、18维零残差动作与2c7b78a旧控制器逐位一致。TRAINING_READY.json和CPU用zero_residual_cpu_check.pt明确native_execution=False，不能用它冒充已训练策略。可导出的实际策略应在当前作业结束后产生policy.pt（新schema full-racket-residual-ppo-v4），checkpoint.pt是保留优化器/归一化/RNG的canonical地面重启点。中间ppo/model_*.pt是非canonical诊断权重，不用作完整resume。CEM入口会拒绝residual actor，防止悄悄丢弃网络。

本次完成后：先核对job SUCCEEDED、8更新、131072实际步、所有loss/梯度有限、数值故障0，以及actor导出同设备对照误差。再经run_gpu_job.py启动scripts/evaluate_full_racket_train_v4.py --checkpoint 当前run/policy.pt --noise none --output 新目录 --headless，固定策略覆盖全部TRAIN128、cap102400，不能直接用训练回合比例称通过。评价入口已用make_rally_controller识别新schema，仍调用原independent_episode与门槛；旧形式锁不得复用于新源码/权重。若作业报错，保留原目录/回执，修具体问题后用新目录；不要因观测延迟重复启动。

CEM35已SUCCEEDED但无明显改善：16候选x32来球=512执行，不是512种。64接触（54框/10空中拍线）、canonical条件恢复3、攻击1、目标1、完整0、NF0。最佳candidate2为5/32空中、3/5条件恢复、1攻击、0目标/完整；它是结构化retarget_fraction0/lead-.01的提案，两条真正拟合参考0/1均未触球。离线拟合2.23cm误差不能视为动态可行。实际.357040617GPUh/409600步，报告CEM35_INNER_STRINGS_RESULT_20261025.json和GENERATION0_SUMMARY已逐回合重算。当前PPO回到已有广分布证据的横向candidate12。

此前失败保留：CEM34航向最佳1/32空中（1/1恢复，不能称覆盖率改善）、0攻击/目标/完整/NF0；CEM33高度最佳5/32、2/5恢复、攻击2、0目标/完整/NF0。报告均已上传。

当前较好的广分布固定策略实测仍是横向candidate12 TRAIN128：空中19/128、条件恢复12/19、攻击7、目标1、完整0/NF0。原正式candidate5 A/B各200仍未通过：A空中24/200、恢复10/24；B空中6/200、恢复1/6；完整0/NF0。所有阈值保持完整≥.50、空中≥.70、条件恢复≥.85、数值0，最终仍需新锁与两面板各200。

最新视频Current_effect_lateral_train128_20261025.mp4，42.66秒/2133帧解码与GitHub下载SHA回验，展示TRAIN23攻击恢复但目标偏远、TRAIN107目标中但速度不足、TRAIN0拍框失败。不是当前PPO效果。62.5MB全TRAIN128轨迹/程序/策略包G07_lateral_TRAIN128_native_and_video_20261025.zip也已上传回验。接触point_velocity_mps是接触步后读数，不当作撞击前速度标定。
