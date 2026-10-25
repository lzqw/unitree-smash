当前唯一GPU0作业：cem35_inner_strings_pose_train32_1150，guard1556243 / child1556257，session98486。源码提交2c7b78a。先核对真实进程、progress和completed/job；禁止重复启动。

继续前重读目标：/home/autodrive/.codex/attachments/55e175f0-9444-4a09-ad37-681e6586ca51/pasted-text-1.txt。只用现有GPU0，无累计GPU/工作窗上限；保留单job故障保护和真实资源计费。原资产、规则、出球/落点/恢复门槛保持。显卡实测RTX5080 16GB；512环境作业初始化后约3.9GB显存，不要误用其他GPU。

CEM35已实际启动：16候选x32个固定TRAIN索引[0,4,...124]=512次执行，不是512种来球。物理62.5us、控制10ms、1代、最多409600控制环境步，guard最长7200秒。回到实际横向candidate12的初始化，航向增益0、高度自适应关闭，算法控制点从UV[0,.10]内移到[0,.07]，不改真实球拍尺寸/质量/碰撞体。打开2,3,7,8,9,10,11,12,13,14,18,25,37坐标搜索，其中8..13是六个臂姿态参数。两个精确参考、八个结构化重定向/挥拍时机候选、六个Gaussian候选。

初始化inner_strings_ready_1145/policy.pt，SHA9f8ceca0089094081126b43a688c449452c4e463d7a5b864b96ce8269bfc7403；candidate_bank.json SHAbf1e6f9742a113094d9d796bac5ed182b12719b1d2926937683f88eefb95a1c5。19个历史TRAIN拍线姿态的离线拟合仍有中位2.23cm、最大2.93cm误差，不能称重定向精确或动态可行。316检查通过；800步冻结输入腿指令逐位一致，臂317步改变。1140是未计名义关节饱和的废弃拟合，不启动它；源代码已按原SHA恢复保存，REJECTED.json保留原因。

CEM34航向方案已SUCCEEDED但效果失败：4x32=128执行，19次接触/18拍框/1空中拍线，条件恢复1，攻击0/目标0/完整0/NF0。最佳candidate3为1/32空中和1/1恢复；两条相同参考均0/32，不将1/1恢复称为覆盖率提升。0.311382262GPUh /102400步。GENERATION0_SUMMARY和CEM34_HEADING_RESULT_20261025.json已用原始行重算。它在同一1.98s前缀的yaw中位-.0636，对照-.1027；姿态变化未转化为任务改善。

CEM33高度方案也未改善：最佳5/32空中、2/5恢复、攻击2、目标0/完整0/NF0；同参数另一批3/32和2/3恢复。0.305926334GPUh /102400步。CEM33_HEIGHT_RESULT_20261025.json保留。

当前较好的广分布实测仍是横向candidate12 TRAIN128：空中19/128、条件恢复12/19、攻击7、目标1、完整0/NF0。同一32种TRAIN索引的历史子集为空中4、条件恢复2、接触20、完整0；只是描述性比较，不是配对因果估计。

正式A/B最后证据仍是candidate5未通过：A空中24/200、条件恢复10/24；B空中6/200、恢复1/6；完整0/NF0。新源码/策略不能复用旧形式锁。

最新视频Current_effect_lateral_train128_20261025.mp4：42.66秒、2133帧，完整解码/GitHub下载SHA回验。展示同一candidate12的TRAIN23攻击恢复但目标偏远、TRAIN107目标命中但速度不足、TRAIN0拍框失败。视频不代表当前CEM35。62.5MB全TRAIN128原始轨迹、程序、策略包G07_lateral_TRAIN128_native_and_video_20261025.zip已上传下载回验。

CEM35完整后运行：/home/autodrive/Isaac/env_isaaclab/bin/python scripts/summarize_full_racket_train_generation_v4.py --run reports/G07_full_racket_v4/cem35_inner_strings_pose_train32_1150 --generation 0 --output reports/G07_full_racket_v4/cem35_inner_strings_pose_train32_1150/GENERATION0_SUMMARY.json。先比较每候选32案例的canonical_air_contacts/conditional_recovery/attack/target/full；若有明显候选，再做TRAIN128或继续同分布优化。没有达到原门槛不称正式通过。接触point_velocity_mps是接触步后读数，不是独立撞击前速度。
