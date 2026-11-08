# 触地误差定位与原生恢复候选对照

2026年11月8日。延续[上一轮实测报告](landing_impact_model_result_20261108.md)。学生基线仍27/128，TRAIN87持续反馈辅助28/128仍有严格隔离异常，尚未晋升教师。

## 冲击误差会保留到控制周期末

本轮完成18次离线诊断：在同一个真实控制区间内，分别从冲击前后不同实测状态开始预测到相同末端。这里只更换离线模型的初态；没有在Isaac中传送或恢复机器人状态，也没有产生新成功示范。

| 事件 | 从控制周期开始预测 | 从冲击前一物理帧预测 | 从冲击结束后实测状态预测 |
|---|---:|---:|---:|
| TRAIN85，298控制步 | 3.01625 rad/s | 3.01955 rad/s | 0.15238 rad/s |
| TRAIN87，303控制步 | 4.22931 rad/s | 4.23323 rad/s | 0.00358 rad/s |

数值为末端最大关节速度误差。TRAIN87冲击分布于两个相邻物理步，表中最后一项使用第二步后的状态；仅取第一步后，误差为0.08400 rad/s。

这些对照支持将主要误差定位到冲击状态跳变，而不是此前整段飞行预测。后移初态同时缩短了预测时长，因此这是定位证据，不是可执行的控制修正或精确因果分解。

## 核对公开TGS方法后的候选测试

读取并锁定NVIDIA PhysX 5.4.0源码，提交`f07ba77f2d8af805824385b4616549974d231a8d`。其接触求解包含累计位移偏置、成对切向冲量、位置迭代与速度迭代；不同于当前预测器一次求解至收敛的接触律。参见[接触求解源码](https://github.com/NVIDIA-Omniverse/PhysX/blob/f07ba77f2d8af805824385b4616549974d231a8d/physx/source/lowleveldynamics/src/DyTGSContactPrep.cpp)和[TGS推进源码](https://github.com/NVIDIA-Omniverse/PhysX/blob/f07ba77f2d8af805824385b4616549974d231a8d/physx/source/lowleveldynamics/src/DyTGSDynamics.cpp)。公开版本与本机二进制的完全等价性没有得到证明。

新增独立有限迭代探针，保留物理参数，只测试预测算法。它使用冻结的步内惯量／雅可比和原模型的有界电机力矩估计，尚未复现持久摩擦锚点、关节求解器全部内部状态或实际接触缓存，不能称为完整TGS移植。

6个已选冲击事件的单步关节速度误差均比原模型下降，但并非都优于上一轮分段模型。例如TRAIN85右前脚事件由9.137降至2.779 rad/s；TRAIN87首次左前脚事件由8.778降至7.851 rad/s。

随后实际连续推进18个10ms区间，全部完成计算。两个主要末端误差仍约3.017、4.231 rad/s，没有实质改善；4个区间还出现候选模型预测速度超限，最大约0.739 rad/s。因此明确不晋升，不接入生产MPC，也不作为专家标签。

这轮进一步说明：接近峰值冲量、单步误差下降，不能代替整段响应与约束一致。下一步不以继续拟合这几个峰值为主要交付。

## 新的原生对照：直接检验被模型拒绝的完整反馈

准备`native_experimental85_all128_20261108_01`：使用此前全身参考搜索中最好的TRAIN85候选。模型只能完成25/520步，仍存在净空／关节约束违反，所以它不是已验收教师。

这次显式将它作为仿真中的实验提案，从地面执行原固定128例；第280控制步之后，只在TRAIN85上运行连续实际状态反馈，直到原回合结束或真实终止。其余127例继续原学生。没有更改评分、物理、执行器、接触阈值或动作限幅。

新增的实验入口与已验证计划入口分开：不伪造520步“可行预测”以通过旧计划加载器，不宣称在线MPC重新求解。它验证的是模型候选的原生执行效果，为后续采样MPC和模型修正提供对照。

启动前已用记录中的真实初态、观测和历史命令执行接口检查：构型、速度、观测、前条命令误差均为零，其他环境动作保持不变。这只是接口检查，不计为原生成功。

GPU0作业设置2400秒与102400环境步上限。结束后自动检查实际命令、原回合、完整成功及未介入案例的逐轨迹一致性；隔离异常照实保留。任何结果都不会自动晋升教师或进入BC正向数据。

本报告初始版本记录的是准备完成，尚无本次原生最终成绩。实时状态与后续证据位于`reports/landing_native_experimental85_preparation_20261108/status.json`和对应原生运行目录。正式G07校验仍因原依赖链和证据不足为FAILED，本次未修改验收契约。

## 程序与证据

- [冲击误差传播诊断](../scripts/audit_impact_error_propagation_v1.py)
- [有限迭代探针](../scripts/temporal_impact_probe_v1.py)、[单步对照](../scripts/compare_temporal_impact_probe_v1.py)、[控制周期对照](../scripts/compare_temporal_control_interval_v1.py)
- [原生实验控制入口v2](../scripts/native_experimental_recovery_v2.py)、[准备程序v2](../scripts/prepare_native_experimental_recovery_v2.py)
- [有界运行与自动审计](../scripts/run_native_experimental_recovery_v1.py)、[实际命令和隔离审计](../scripts/audit_native_experimental_recovery_v1.py)
- [紧凑结果与输入哈希](landing_impact_localization_native_candidate_20261108.json)

## 首次启动失败与独立进程修复

首个原生作业已终止，保留在`native_experimental85_all128_20261108_01`。环境建立后、执行控制回合前，适配器调用离线`state_from_row()`时发现全局`_WORKER`为空，抛出`KeyError: controller`。此前准备进程恰好初始化过搜索器，掩盖了这个进程依赖。不能把它解释为控制器落地失败，也没有生成新成功率。

v2由自身机器人模型与记录构型直接构造`PublicState`，不读取离线搜索全局对象。新增[独立进程检查](../scripts/check_native_experimental_recovery_v2.py)：全局worker始终为空，正确查询通过；将查询位置改动1mm时按原容差拒绝，未放宽匹配条件。

新的完整作业目录为`native_experimental85_all128_20261108_02`，准备与运行状态在`landing_native_experimental85_preparation_20261108_02`。旧代码、失败日志和用量保留；不覆盖或重启原作业目录。首个作业记录GPU用量0.002706小时；其102400是失败无收据时的保守计费步数，不是已经执行完的训练步数。
