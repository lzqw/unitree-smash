# Recovery 原生动作响应检查：已完成采集，未获训练准入

基底 0e417225f2b75c5978e6f72856d0f423d79a31c8，保留全部既有修改、模型及失败结果。GPU原生作业 supermimic-native-recovery-t302-20261118.service 正常结束，退出0。Actor/World/Critic优化器更新均为0。

## 原生运行与合同

原128环境、seed882072、causal恢复门控、cuda:0、headless。四个独立从地面相同前缀重放：nominal_a、nominal_b、minus、plus；每个303控制tick，总1212tick／155136环境转移。固定在tick302对FL_hip_joint请求±0.02rad扰动，原mixer和真实限幅保留，实际最终命令正负差约0.016rad。不是800tick正式评分，没有生成完成评分文件。

目标时刻在新实验之前依据96个训练lane首次着地事件覆盖选定；2个预留首次着地lane为7和37。案例仅作诊断，没有案例动作规则。两次nominal完整起点、Actor、执行器记忆、命令和下一状态完全相同。四分支全起点／代码／策略／任务／物理身份一致，最终源码哈希未变，退出码全部0。status、独立SHA256、shutdown记录在result_summary.json及原始目录。

## 模型与原生结果

对同一真实当前状态、记忆和最终命令进行CPU只读对照，所有预测和梯度有限。原模型SHA3045993066c53142aa337bea3d5f2e10ee28effdbe3201dc1659db5fba1204c0；此前完成的原生响应校准模型SHAfaf73188aa54d76b2782b86c50f9b3652f67e6e0a346c21c7d7dd25e9324929b。

| 32个留出lane | 原模型 | 响应校准模型 |
|---|---:|---:|
| 机身角速度响应余弦中位数 | 0.3501 | 0.5023 |
| 机身角速度响应相对误差中位数 | 0.9761 | 0.9435 |
| 机身角速度负余弦案例 | 7/32 | 5/32 |
| 关节位置响应余弦中位数 | 0.5256 | 0.6160 |
| 关节速度响应余弦中位数 | 0.6721 | 0.6617 |

预声明首次着地留出lane7：校准模型角速度响应cos0.71746，幅值比0.22923；lane37：cos−0.01604，幅值比0.91546。后者接近正交，不能夸大为强反向证据，但也不能视为可信方向。中点自动微分投影与上述方向一致；非零导数不等于真实方向正确。

记录的边界足端／身体接触模式124例相同、4例变化。模式变化只解释为有限扰动结果，不声称光滑Jacobian；边界模式相同也不证明全部1ms子步模式相同。单一时刻、单一方向不能证明完整Recovery质量。没有依据此次结果放宽或拟合准入阈值。

## 工程与训练状态

原生四分支运行及只读对照均退出0，数据完整；物理模型可信性尚未验收。对照命令：

```bash
CUDA_VISIBLE_DEVICES='' QUADSMASH_PYTHON=/home/autodrive/Isaac/env_isaaclab/bin/python bash scripts/python.sh reports/supermimic_admission256_20261118/recovery_probe/compare_models.py
```

comparison.json保留全部128例、最终命令、模型响应、自动微分投影及模式分组。comparison.log保留输出；result_summary.json保留所有分支退出与哈希。未启动256次Actor训练、未新增正式成绩、未把旧TensorBoard当作本轮训练。

Approach仍有既有独立球速方向失败。第四轮球门控模型修订代码已准备，尚待此前提出的额外修订预算授权；本次Recovery证据不会自动许可新增模型训练。不能把有限模型拟合失败表述为硬件或任务无法恢复。
