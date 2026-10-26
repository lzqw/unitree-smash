# 后续原生候选的工作记录（尚未实现）

当前原生所有者1827656运行7个拍线瞄准点，419份输入保持到其实际退出。新控制器/脚本应先在隔离worktree开发，只在所有者真正退出后安装新路径。普通算法调整已有用户授权，不需要再问许可；资产、原2×2米评分和GPU0单作业保护保持。

建议先检查实际参考点实验的最佳冷评估。若仍未达到原50/70/85、NF0，准备一个小型腕部/引拍姿态原生比较，而不是继续延长已退化的48更新课程。

- 数据应优先使用所选来源固定128的contacts.json：它含精确接触时刻全部关节角/速度、28个身体姿态、根部COM位置/速度/角速度及法兰/触点速度，优于50Hz近邻帧。仅从TRAIN选择一个已完整成功的参考；生成的控制种子是共享常数，不能以案例ID查表。
- TRAIN29的实际腕部角度替换为[q4−pi,−q5,q6−pi]并适当wrap后仍在原限位内，拍头差约0.18毫米，目标限速下界0.4404→0.3108秒。它没有执行过。准备至少保留未改来源、实际成功姿态不镜像、镜像成功姿态三个候选，才能区分预置姿态与腕部分支的作用。
- 新wrapper可调用现有contact_aim工厂以兼容body/第三层残差/Aim来源，保持所有源残差actor/JIT/RMS字节。只改实例的teacher及reference参数，不改嵌入checkpoint。旧FullRacketTeacher有joint_preparation、face_feedback、lateral_residual_feedback和原5rad/s限速；须先检查这些模块的具体约束，保留硬件/物理防护。
- 新hit_q应直接设teacher.arm，并把运行参数8:14归零，避免旧的大关节偏移重复叠加。重算完整球拍pitch_basis、reference Jacobian/inverse；任何reset后重装运行参数，因为源body.reset会恢复旧population。改变步行位置用参数1的目标偏移，从原ground0自主走过去，不能改初始化root。
- 改速度前先读取现有arm_velocity工具的参数索引。可用精确原生身体twist和完整拍触点Jacobian做限幅关节速度初值；建议比较原实测触点速度与期望[0.8,0,−0.3]/[0.8,0,−1.2]米/秒。5rad/s是软件目标变化限制，不能把这个静态速度初值当作实际速度/力矩可行证明。保留原球拍质量和omega×杠杆臂。
- 设置新的nominal_body_pitch时要和用于IK的真实参考身体姿态一致。body37项和teacher.normal_world应一致，避免额外法向偏置；读取实现后再改。原teacher.velocity_world为[1.5,0,−0.3]，nominal_pitch为0；原成功触点实测为[0.8231,−1.6163,−0.1040]，已有明显的侧向运动。
- 原生搜索可把少量候选各128例组合成一批（不超过已验证2048环境），分组控制、保留全局env_id和实际episode_id，正常地面reset。保留每例真实姿态与前两例公共输入动作，独立重算后选优，再单独冷128确认。纯CPU命令参考应逐项等于旧来源，其他候选需有限、边界与5rad/s检查。
- 新公开策略应使用独立schema并显式记录runtime overrides、source SHA、实际原生证据与performance_inherited=False。已有隔离正式入口只支持body/第三层/Aim，后续新族需要在真正满足准备门槛前补齐工厂与锁定支持；不能提前打开A/B内容或扩大成功定义。
