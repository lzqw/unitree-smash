# G07 完整球拍 v4-r3：真实来球适配进行中

目标文件55e175f0...已读。仅GPU0，无累计GPUh/工作窗/日历截止；所有运行通过run_gpu_job单作业、有限每job超时与交互上限，保留历史账本，不创建云资源。不要询问旧截止，不继承旧G00-G06短拍板通过。

资产assets/quadsmash_full_racket_v4_r3.usd（c2264fce...）：675mm整拍、452mm法兰至拍线中心、153g整拍+夹具、总20.709991kg。球场13.4×6.1m，悬空网6.1m宽/0.76m深，中1.524m、侧1.55m。r1漏碰/r2夹具穿腕失败均保留；r3原生接触/直立地面自碰撞与质量COM惯量回归成功。详见MODEL_RULES.json和前版NEXT_before_fine_adaptation.txt。不是新策略验收通过。

冻结协议configs/g07_full_racket_eval_v4_r3.json；A/B各200、训练128来球及50%/70%/85%/NF0门槛未改变。拦截平面1.10m是来球生成/通用命令参考，不是禁止其它己方合法拦截位置的规则。策略可以根据当前公开来球预测选择x1.38m；必须仍通过实际接触点、全身越网、网体/柱和首次落点检查。

本轮真实证据：
1. rally_probe_r3_01（session22673已终态FAILED/exit-9，不再运行）诊断提前停止，非完整8秒评估。一个原始回合0.9s前摔落；其余三个首回合到tick500均已无触球且回合死亡，不能在后续重新成功。SIGTERM被Isaac处理后循环未退出，49秒后仅对核实的本项目child SIGKILL，原始partial与early_stop_snapshot保留。实际749.541175s/0.208205882GPUh；账本保守计步3200，记录的完整部分至少500×4步，不冒称精确实际步数。没有完整恢复/NF总数。
2. 公开观测反推的预测到达点与实际来球接近，问题主要在拍头低/偏侧。partial_failure_analysis.json记录每案；不是数学不可行证明。
3. reach_cem_fine_feedback01（session83419已SUCCEEDED）：62.5us物理、1ms跳跃指标、10ms控制，2代×32×6秒，644.222310s/0.178950642GPUh，38400控制步。64候选中6次跳跃恢复，3次没有网接触或身体过平面，40次碰网。最佳仍为原参考，虚拟球点[1.20645,-0.05464,1.57923]m，速度[-0.14289,0.30576,-0.44402]m/s；未满足高位/速度目标。其旧无球材质未显式与BallEnv对齐，不能当正式材质一致性验证；只作课程结果，CONTACT_MATERIAL_SCOPE.json解释。
4. 新Torch批量URDF Jacobian与独立SciPy有限差分一致；2000个实际公开状态的拍头位置误差<=3.24e-6m、法向误差<=2.43e-6；关闭新反馈时动作差0。32项针对性CPU测试通过，不等于策略通过。
5. 修正合法近网球事件：首接触点在己方而球心已在对面时，按球后缘实际通过网面记录，并允许同次挥拍中先过网后完全释放；完整仍要求攻击出球/首次目标落点/恢复/无违规。RULE_FIX_NEAR_NET.json与正反逻辑测试保留；该边界尚未专门原生验证。

上一近网任务reach_near_net_backface01已终态FAILED（有意诊断停止，非程序异常）：完成首代64候选，62触网/30身体越面，没有同时安全的候选；下一代449tick后响应SIGTERM保留interrupted.json和generation_start.pt。实际576.586533s/0.160162926GPUh；账本153600步为无完成凭据时保守上限，实际至少(600+449)×64=67136控制步，不能冒称四代已完成。

当前活跃GPU任务：reach_near_net_backface02，exec session26054，guard PID1317741/child PID1317753，started_at2026-10-24T03:37:03.211119+00:00；上限2700s/153600控制步。相同r3资产/62.5us/接触契约/冻结来球，走位均值退到0.85m、速度0.65m/s，优先选择无触网/无身体越面且虚拟点在己方的候选（不是删评测案例）。检查同一进程/progress，勿重复启动。job.json只在guard终态写，运行时看GPU_LEDGER最后一条。

当前用户要求生成效果视频：scripts/render_full_racket_r3_effect.py，以CPU llvmpipe渲染实际r3全身姿态；无球恢复样例+真实来球漏球partial+近网01触网失败，侧视/俯视/慢放。视频已生成reports/G07_full_racket_v4/video/Current_effect_r3_20261024.mp4，38.12s/1906frames/50fps/1280×720，SHA2cbce97da8ab037d43585425edccadbafa1fdf1b281c6c5e97be7e6011dd36f7，完整解码无错误。已上传同一v4进度release，视频和24.17MB原始证据zip（58文件，不含活跃02）的5个附件均核对Github大小与SHA256。不要宣称触球或正式成绩。

当前方法：同一真实拍线面的反侧，实际uv=(0,0.09)m上部拍线点（仍在原边界内），预测拦截x1.38m，目标1.90m/点速[3.2,0,-1]；CPU几何种子ik_near_net_upper_strings01只是初始化。24参数含走位、准备、腿时序/前后差分、挥拍/收拍、公开状态Cartesian反馈。源权重保留，硬件/质量/网/来球未改。--warm-start使用旧腿骨干但重测全部真实fitness，不能当旧成绩通过。首轮关节参考和反馈样例预定记录前2lane，所有候选/失败均保留。

后续必须连续推进：先读此任务真实结果→保留高位/速度与可恢复候选→执行真实球训练库开发试验及联合优化/学习，不在无球课程停止。正式400前需原生验证扰动reset和评估器；scripts/lock_full_racket_policy_v4.py、evaluate_full_racket_v4.py、aggregate_full_racket_eval_v4.py仅已实现/编译及7项CPU完整性测试，尚未原生运行、没有正式成绩。不要用正式面板选候选或删例。最后冻结代码/权重跑A/B200，生成实际触球与首次落点侧视/俯视视频及完整比较，再做完整目标审计。当前仍未完成合法跳跃扣杀。

代码分支g07-full-racket-court-v4-20261024。已有初版进度附件：https://github.com/lzqw/unitree-smash/releases/tag/g07-full-racket-v4-progress-20261024 。后续报告ADAPTATION_PROGRESS_20261024.json包括本轮费用、失败、限制和活跃handle。38秒旧进度视频包含r1站立与旧短拍板对照，不是r3扣杀。


本轮后续（2026-10-24T03:52Z）实质进展：reach_near_net_backface02前2代128次完成；安全候选15→27，第二代1个无触网/无身体越面并恢复的候选（lane17，腾空0.509997s、稳定2.123029s、body ground0N）。其虚拟球心[1.384733,0.074827,1.711490]m、点速[1.817642,-0.821297,-0.098624]m/s、面法向[0.668906,-0.057400,-0.741128]，仍不构成高位攻击触球；不要宣称完成扣杀。02四代任务继续，未停止或重复启动。

已启动依赖队列native_followup_sequence01，exec session30733、CPU PID1320916（不是第二个GPU任务），state.json是权威状态。当前等待同一个02 guard1317741/child1317753成功终态；然后串行调用run_gpu_job：
1. reset_eval_native01，8个TRAIN输入、6控制tick、原生核对B扰动实际q/qd/XY/yaw、公开观测、局部reset隔离，最多240s/48控制-env步；不是正式评测。
2. 若1成功，ball_cem_smoke01，4候选×TRAIN indices[1,65]×1代×8s，62.5us，最多2100s/6400控制-env步；使用02最终最佳腿参数、CPU新平拍面ik_flat_face_fast01初始化，真实拍线距离/接触/出球/目标/恢复联合优化26参数（增加己方拦截x和时机残差）。CPU种子不是执行证据；新种子的初始时间校准为0假设，需实际fitness学习，不能沿用旧种子已验证时序的说法。

队列固定了当前quadsmash和两个脚本/核心配置源码哈希；如在其等待时修改这些源码，它会停止待检查，不会默默运行另一版。先查sequence_state和GPU_LEDGER真实进程，再决定下一条命令；不要与队列重复启动任务。后续如任一步FAILED，保留error/stdout，核查并修复，不能把等待失败/timeout当终态成功。scripts/search_full_racket_ball_v4.py、smoke_full_racket_eval_reset_v4.py、共享full_racket_eval_env_v4.py目前仅编译，尚未原生执行。

控制修订：旧教师进入腿恢复时直接把机械臂命令切到default，CPU同一公开状态在边界前后20us的目标最大跳变0.295982rad。新模式将完整的计划收拍带入腿恢复，跳变8.123e-5rad；只在新ball训练seed.smooth_recovery_arm=True启用，旧checkpoint默认行为保留。recovery_arm_continuity_cpu01.json有不可变population_000.pt来源hash。尚无原生恢复改善结论。公开观测规划测试核对4个己方拦截平面/时机残差，默认扩展动作差0；32项针对性CPU测试再次通过，仍不是策略验收。
