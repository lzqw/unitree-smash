# 当前唯一GPU：ball_recovery_cem01，实际联合训练继续

仍须读原目标 /home/autodrive/.codex/attachments/55e175f0-9444-4a09-ad37-681e6586ca51/pasted-text-1.txt。完整目标不缩减：真实完整球拍/标准悬空网/己方合法空中拍线+攻击>=8m/s、vx>0、vz<=-1+真实越网首次目标落点+同回合无bodyground稳定.8s，再冻结A/B各200按worstpanel .5/.7/.85/NF0与全部Github/实际成功失败视频/旧新比较。当前无完整成功，绝不标完成。仅GPU0，无累计GPUh/总工作窗/截止，有限每job保护。

活跃exec53083、guard1334795/child1334807，ball_recovery_cem01；32候选×TRAIN[1,65]=64环境×2代，每回合最多原8s、62.5us物理/10ms控制。单job5400s/102400控制-env步cap，warm-start recovery_curriculum_initialization01/policy.pt，不继承旧fitness。先查同一PID/progress/error/job.json或poll53083；观察超时不是终止，不重复启动。GPU_LEDGER/RUNS由活跃guard持有，勿手改。

当前新课程--continue-after-miss：仅TRAIN允许球已经首次落地漏失的回合继续观察同一跳跃的落地恢复，回合失败/原native事件标志保持，后续弹跳不能变成功；触网等rallyfault/bodyground/fallen仍结束。加入实际qualified landing/稳定恢复奖励、实际bodyground最终扣分，正式FrozenPanelEnv/evaluator仍完整原8s和原成功定义，不受课程类影响。--push-search-range .30仅扩大前后腿位置目标参数的搜索区间，不改变实际关节/速度/力矩限幅。新采样/目标/源码必须warm-start重测，不能直接继承旧resume fitness。

初始化来源timing_grid01 candidate8（phase1.38/armgain.85/servo.4）与push_grid01中较低转速push组合[+.15,-.15]；这个组合此前未执行，不称可恢复或已触球。旧timing candidate8的TRAIN1无bodyground直到漏球，但TRAIN65实际bodyground1658.6N，已诚实记INITIALIZATION.json。当前新job必须实际测两案例。旧弱触球权重完整保留；若该课程无进展，按实际state继续角速度/姿态反馈、反作用协调、其他已证实无球恢复seed或残差/轨迹优化，不固守失败教师。

刚完成ball_push_grid01/session65283（guard1332513/child1332525均退出）：36首回合/15接触（8空中拍线/7拍框）、0攻击/0越网/0完整/NF0。实际424.246771s/.117846325GPUh/12456步，保守账本cap28800。部分前后腿偏置在同一precontact t2.74降低根角速度，但实际触点vx仍负，不能把更多触球当有效扣杀。完整TRAIN_RESULT_SUMMARY.json、36行episodes.csv、全部episode/contact/window/public/weights与source保留。新TRAIN末状态完整28刚体/球已实际记录，不伪称精确首次碰撞子步。

本轮还补齐10个组成USD依赖层（4个仓库外Go2/6个仓库内完整资产与PiPER），configs/full_racket_usd_dependencies_v4.json固定全部size/SHA。外部文件改变或缺失会使锁/评测失败；锁归档包含全部依赖文件。34项CPU测试通过，policy_lock_integrity_smoke01实际锁写入/loader/zip10层hash验证通过；只弱策略完整性烟测，非最终选型/非正式成绩。当前recoveryjob运行中的Isaac USD库已实际核对10层GetUsedLayers与hash匹配，USD_COMPOSITION_NATIVE_CHECK_20261024.json。独立CPU工具未找到内置OmniPBR.mdl的运行时shader路径，标明视觉shader依赖，不是物理层缺失。

此前28.94秒真实触球视频已经上传并核SHA：
https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/Current_effect_air_contact_20261024_0450.mp4
侧视俯视/慢放/精确接触快照/拍框首落点，0完整成功。旧38秒视频保留。刚完成pushgrid与依赖锁/本轮程序已上传Github并核大小SHA，活跃recovery数据未作为完成附件。增量链接：https://github.com/lzqw/unitree-smash/releases/download/g07-full-racket-v4-progress-20261024/G07_coordination_recovery_20261024_0502.zip。

后续仍须扩大TRAIN128与原扰动、验证原生近网整拍/随挥全回己方的边界（旧racket角点AABB代理偏保守、followthrough返回用above谓词待定向修正验证）、冻结真实最佳源码权重、原A/B200独立、真正完整成功/失败视频和旧新能力比较。原r3资产675mm整拍/451.8mm法兰至拍线中心/153g拍夹具/20.709991kg总重、执行器、球、原train128/A200/B200哈希/原门槛都不变。不能从CPU测试/静态IK/failed sampling宣称完整成功或不可行。不得改preexisting status/G07_RESOURCE_STOP.json，不停止无关MPS/GPU任务。

更早上下文见NEXT_before_recovery_cem_0502.txt和此前NEXT快照。
