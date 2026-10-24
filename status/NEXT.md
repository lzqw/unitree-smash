# 唯一活跃GPU：ball_recovery_cem01第二代，原目标仍未达成

继续前必须读 /home/autodrive/.codex/attachments/55e175f0-9444-4a09-ad37-681e6586ca51/pasted-text-1.txt。仅GPU0，无累计GPUh/总工作窗/截止，有限单job保护，历史失败/费用均保留。完整目标不可缩减：真实完整球拍/标准悬空网/己方实际合法空中拍线+原攻击>=8m/s、vx>0、vz<=-1+真实越网首次目标落点+同回合无bodyground稳定.8s，再原A/B200各面板worst .5/.7/.85/NF0、成功失败实际侧视俯视视频与旧新能力比较、Github完整交付。当前完整成功仍0，不能标完成或说不可行。

当前同一exec53083、guard1334795/child1334807，ball_recovery_cem01；32候选×TRAIN[1,65]=64环境×2代，62.5us物理/10ms控制，每回合原8s。单job5400s/102400控制-env步cap。先ps/poll同一handle/progress/error/job.json；观察超时不等于退出，不重复启动。GPU_LEDGER/RUNS由活跃guard持有勿手改。当前已实际进入generation1（第二代），第一代完整800ticks/51200步；整体GPU费用待guard终态结算，不捏造每代GPUh。

第一代结果：64次候选执行（含重复参考）0触球/0完整/NF0，3次qualified jump+稳定恢复，均在漏球后同一回合，不能称成功回球。env2/candidate1/TRAIN1：air.477、stable3.9299s；env8/candidate4/TRAIN1 air.466、stable3.6269s；env13/candidate6/TRAIN65 air.405、stable3.8719s。所有64失败也保留。recovery_generation0_snapshot01拷贝并逐文件SHA核对了第一代不可变raw/权重/源码，GENERATION.json说明job仍未完成。第一代至少一组两案例均无bodyground/无rallyfault，但一例稳定仅.554s，原恢复门槛未降低。第二代在同一旧源码进程继续实际CEM。

当前运行源码是启动时snapshot；下列新代码仅已编译/CPU验证，绝不冒称当前job已用：
1) RacketPlaneSupport对声明USD圆柱/胶囊/62拍线盒求真实support，整支球拍回己方才结束随挥；不是仅“不高于网顶”。CPU几何/规则38项通过；新增课程不复活回合/混合robot fault测试后当前40项通过，96个USD碰撞primitive/308080表面采样/64随机朝向核对，最大采样间隙7.285um、无低估；这不是原生运动验收。
2) Native PhysX overlap_box查询对手侧、实际64网顶分段以上的体积，并只接受该环境实际球拍碰撞shape（Clamp仍属机器人）。AABB只作query范围，不再判材质越网；1um数值容差显式记录。cfg开启scene query，不加力、不改变asset/质量/执行器/球/来球/门槛。当前未原生执行，必须按NEAR_NET_RULE_CHANGE_0526.json的3条精确guard argv依次运行near_net_own_contact01、near_net_empty_corner01、near_net_prehit_invasion01；每个nonground/fixed-base诊断仅1控制步、max180s、cap1，不能称策略成功。必须等活跃GPU旧job终态后执行，不并发。失败保留新目录修复后测，不覆盖。
3) 新TRAIN/formal logger将保存全部lane完整回合50Hz姿态/球/clock/first-valid，另存原生接触/终止快照。当前旧job仅首2lane连续全回合，all-lane窗口2.2..3.4；env2/8/13的后续恢复不能拼接别的lane伪造视频。新每代清除plane事件/重置local episode id0；当前旧jobgen1的global episode id可能>0，优化仍按每env第一条记录，勿误滤只episode0。

接着必须直接继续真实适配，别停在边界审计：3个原生短回归过后，以当前ball_recovery_cem01/best_candidate.pt作warm-start，--ik-seeds reports/G07_full_racket_v4/ik_recorded_upper_strings04/seeds.json，继续--continue-after-miss --push-search-range .30、32候选×[1,65]×2代，有限5400s/102400cap新目录ball_upper_strings_cem01。新源码/目标/控制用warm重测，不能直接resume旧fitness；若要旧optimizer精确resume须先恢复其完整source snapshot。

upper_strings04是可复现CPU proposal，不是成功：scripts/retarget_full_racket_recorded_pose_v4.py从immutable TRAIN_SOURCE_SNAPSHOT拟合低joint1分支，uvz+.09仍在原真实拍线内（没有放大拍面）。native gen0/env36在t2.74的姿态，静态点误差1.5e-7m/normal1.2e-6，局部带根运动的速度投影[1.9723,-.6281,-.8536]m/s；实际受新arm反作用会改变，不能拿预测冒充真实v_hit。源env36后来恢复失败，下一步取实际恢复更好的新腿策略并重新测耦合动力学。seed新增nominal projected pitch.23335、leg_event_time_gain0（arm仍public ball causal timing，腿时钟固定，旧种子默认1不变）；实际outgoing8m/s攻击门槛完全未改。旧01/02/03都是未执行CPU草稿保留，以程序生成04为下一实测候选。CPU几何失败/搜索失败不代表任务不可行。

后续若该teacher无实质进展，按真实状态推进姿态/角速度反馈、反作用协调、轨迹优化或残差学习，不固守教师。扩大TRAIN128与原扰动，冻结真实最佳源码/权重（包括10层外部USD依赖，前轮已原生核对），独立原A/B200，再实际完整成功/失败全身视频和旧新比较。原r3资产c2264fce...、675mm整拍/451.8mm法兰至拍线中心/153g拍夹具/20.709991kg总重及全部原物理hash与门槛均不变。

已发布最新实际触球视频28.94秒：
https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_air_contact_20261024_0450.mp4
此前36次pushgrid的8拍线/7拍框/0完整原始证据、程序与10层USD完整性归档已经GitHub上传并核size/SHA。当前第一代snapshot与新代码已经GitHub上传并核size/SHA；第二代未完不当完整训练附件。已另保留gen1/env62/episode2的首回合部分证据：真实合法空中拍线点高1.702071m、out4.950732m/s/vz-1.0492，实际shuttle_net_contact和net_blocked_fault，0complete/NF0。policy_evidence.pt将旧初始化training_result单列，未虚称整对案例完成。历史上下文NEXT_before_near_net_change_0526.txt和此前NEXT文件。不能编辑preexisting status/G07_RESOURCE_STOP.json，不能停止无关MPS/GPU任务。

未来课程v3补充（当前运行仍v2）：--continue-after-miss还允许shuttle-only的bad_return/net_blocked后继续观察同一跳跃，rally始终dead/fault、complete始终False，robot-rule/bodyground/fallen仍提前终止。CPU40测试包括“不能复活失败回合/不能用球故障掩盖同时robot fault”。新合同ball-with-jump-recovery-v3，必须warm重测。CONTINUATION_UPDATE_0530.json及GitHub最新分支记录完整来源。
第一代包：https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_recovery_gen0_and_rule_code_20261024_0528.zip
第二代接触部分证据/当前v3程序：https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_recovery_contact_and_curriculum_v3_20261024_0532.zip
