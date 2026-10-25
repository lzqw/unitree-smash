当前唯一GPU0作业：cem34_walk_heading_train32_1120，guard1551980 / child1551994，session5222。源码提交c92342a。先查真实进程、progress、completed/job；不要重复启动。

继续前重读目标：/home/autodrive/.codex/attachments/55e175f0-9444-4a09-ad37-681e6586ca51/pasted-text-1.txt。无总GPU/工作窗上限，仅GPU0；每个job保留故障保护和真实计费。原物理、规则与成功门槛保持。

CEM34：以此前实际横向candidate12为初始化，关闭高度自适应，启用当前公开重力/已知+X方向恢复出的航向反馈gain1.5。4候选x32个均匀TRAIN索引[0,4,...124]，1代，最多102400控制环境步。准备策略walk_heading_ready_1115/policy.pt（未执行初始化文件），SHA76a96f7146d9f996f762a1ffa8882019f758893ed3f2fd6c7682763375a2de25。310检查通过，CPU800步默认/零增益逐位旧版一致，启用只在准备期改变冻结输入上的腿指令；闭环物理表现需实测。

CEM33高度规划已SUCCEEDED：4候选x32种来球=128次执行，不是128种。最佳candidate0空中5/32、条件恢复2/5、攻击2、目标0、完整0、NF0；同参数candidate1空中3/32、条件恢复2/3。实际0.305926334GPUh /102400步。GENERATION0_SUMMARY逐候选重算一致，CEM33_HEIGHT_RESULT_20261025.json记录失败。无证据说明分布改善，不重跑正式A/B。旧queue1546233已退出并SUCCEEDED。

当前最佳广分布实测仍是横向candidate12的TRAIN128：空中19/128、条件恢复12/19、攻击7、目标1、完整0、NF0。原始行独立重算一致。相同CEM33/CEM34的32种TRAIN索引在此历史执行中为空中4、条件恢复2、接触20、完整0；这是描述性比较，不能称因果提升。

正式A/B最后证据为candidate5未通过：A空中24/200、条件恢复10/24；B空中6/200、恢复1/6；完整0/NF0。旧形式锁不可用于新策略或新源码。

最新视频：Current_effect_lateral_train128_20261025.mp4（42.66秒、2133帧，完整解码与GitHub下载SHA回验）。同一candidate12的TRAIN23攻击恢复但目标偏远、TRAIN107目标命中但速度不足、TRAIN0拍框失败。真实侧视/俯视/接触/首落，不代表CEM34效果。62.5MB全TRAIN128原始轨迹/程序/精确策略包G07_lateral_TRAIN128_native_and_video_20261025.zip也上传下载回验，SHA64cfa0cc757374b95f27354490963ba62f2996d60d40a2d76b0b789e2598f19f。

完成CEM34后运行：/home/autodrive/Isaac/env_isaaclab/bin/python scripts/summarize_full_racket_train_generation_v4.py --run reports/G07_full_racket_v4/cem34_walk_heading_train32_1120 --generation 0 --output reports/G07_full_racket_v4/cem34_walk_heading_train32_1120/GENERATION0_SUMMARY.json。先看每候选32案例的canonical_air_contacts/conditional_recovery/target/full。若有明显候选，再做完整TRAIN128或扰动诊断；未达到原门槛不称正式通过。接触point_velocity_mps是接触步后读数，不能当作独立撞击前速度标定。
