# G07 完整球拍 r3：真实来球联合训练正在执行

目标文件55e175f0-9444-4a09-ad37-681e6586ca51/pasted-text-1.txt必须读。仅GPU0，无累计GPUh/总工作窗/截止；每job有限保护，所有费用和失败保留。禁止修改资产/硬件/网/冻结来球/门槛来通过，不继承旧G00–G06成绩。大目标仍是合法空中拍线击球+攻击出球+越网首次目标落点+同回合稳定恢复，然后冻结独立A/B各200（50%/70%/85%/NF0）和完整交付。当前没有新正式成绩，不标完成。

实际当前作业：ball_cem_smoke01，guard PID1323147/child1323156，started_at2026-10-24T03:59:09.882873+00:00，来自native_followup_sequence01（CPU PID1320916、exec session30733）。先查sequence的state.json、同一PID/progress/stdout、GPU_LEDGER最后条；不是另起任务。job.json只在终态写。最多2100s/6400控制-env步，4候选×TRAIN indices[1,65]×1代×8秒，62.5us物理/10ms策略。已运行至200ticks，8个首回合均未提前结束；尚无触球/完整结果。不要因等待超时重启。

已完成：reach_near_net_backface02四代64共256次执行（含精英重测），1311.131635s/0.364203232GPUh、153600控制-env步，guard SUCCEEDED。安全（无网/无非球拍越面/虚拟点己方）88次，安全跳跃恢复8次，不能当8个不同策略或真实击球。最佳虚拟球心[1.379143,0.039516,1.741971]m、点速[1.744774,-0.809993,-0.219007]m/s、腾空0.503997s、稳定1.912033s、bodyground0N，面法向仍很向下，未具攻击击球条件。闭环权重best_candidate.pt和四代全部population/episodes/public姿态均保留。原session26054已终态，不再轮询它。

队列的reset_eval_native01已SUCCEEDED：8个TRAIN输入、B扰动q/qd/XY/yaw、公开观测一致性、局部reset不影响其它环境；13.490756s/0.003747432GPUh、48控制-env步。初始XY最大误差6.706e-7m，其它写入误差0。此版本核对writer/ArticulationData与观测，并跑6步真实物理；后来在脚本加了直接PhysX view独立读取，目前还未原生运行，不能把旧烟测扩大为这项已通过。下一次空闲可用新目录reset_eval_native02跑该扩展（240s/48步），不要覆盖01。

当前ball训练用02最终最佳腿参数+CPU ik_flat_face_fast01初始化（拍线中心、正面、x1.12/z1.78，根假设x.65/z.72），26参数包含己方拦截平面和时机残差，真实接触/出球/目标/恢复评分。新几何种子初始时序校准0是初始化假设，不是已测时序。smooth_recovery_arm=True仅用于新ball种子，保持收拍连续；旧checkpoint默认False行为保留。CPU边界命令跳变0.295982→8.123e-5rad，只是命令诊断，尚无原生改善结论。

CPU只读滞后拟合measured_arm_lag_cpu01.json显示本次安全飞行多数关节约58–77ms表观响应，支持现有60ms前馈量级；是单轨迹经验拟合，不是硬件标定。又生成两组未执行几何种子ik_flat_upper_fast01（正面，原拍线上部uvz.09）/ik_back_flat_upper_fast01（反面，几何误差更大），都没有改变拍面边界、质量或来球。可按实际Ball结果决定后续，不能用CPU IK宣称可执行/不可行。

本轮在当前Ball进程已完成模块导入并写generation_start.pt后，增强了未来版本的中途轨迹/距离/首回合接触保存、所有触球episode/event编号，以及重置smoke的独立PhysX view核对。这些不改变物理/策略判定；当前活跃进程仍使用source_snapshot.zip中旧版本。脚本源码hash变更后不能直接resume继承旧fitness，用best_candidate warm-start重测，或用旧快照恢复匹配源码。队列已进入最后一个任务，不再有待启动步骤；结束后仍看state/receipt。

r3资产c2264fce...，675mm整拍、452mm法兰到拍线中心、153g拍+夹具、总20.709991kg；13.4×6.1m球场、6.1m宽0.76m深悬空网，中1.524/侧1.55m；r1漏碰/r2夹具穿腕与旧失败均保留。正式协议configs/g07_full_racket_eval_v4_r3.json，A/B200和train128哈希未变。正式评估器已实现/CPU测试，扰动reset原生初步通过，但独立400仍未跑。sources/MODEL_RULES和前版NEXT快照保留完整模型依据、失败与费用。

后续：收当前Ball真实结果→处理实际报错/漏球/触网原因并继续实际训练，不在无球/文档停止；需要时换正面上部拍线种子或优化准备/时机/收拍，不改门槛；验证评估器/冻结代码和策略后跑400；生成实际成功及失败侧视/俯视/触点/首落点视频，补旧新能力比较。不得从少量失败或静态高度给出物理不可行断言。GPU账本由运行guard独占写，勿手改。

已交付38.12秒当前r3视频和24.17MB原轨迹证据，已核Github大小SHA；链接见README。程序分支g07-full-racket-court-v4-20261024，当前HEAD cd62241；后续增量报告NATIVE_PROGRESS_20261024_0400.json记录完整256结果、reset费用及活跃Ball指针。GitHub附件归档排除活跃ball目录，不冒称结果已完成。

正式冻结前尚需原生近网边界案例：CPU合法接触/有限球心过网测试不等于整拍真实边界已原生验证；核查racket包围点代理对prehit入侵与“完整回己方后再次过网”判断的影响，必要时改用真实碰撞形状支持界并保留失败。不能修改BWF接触点规则或删正式案例。
