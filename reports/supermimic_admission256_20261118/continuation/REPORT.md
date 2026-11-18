# 继续推进：监督覆盖、保留集、成功参考与原生探针

本次在0e417225f2b75c5978e6f72856d0f423d79a31c8及当前dirty代码上继续修复，未覆盖旧候选或用户历史文件。目标仍为通过全部准入之后启动原128环境、256次实际Actor更新，并提供持久运行与TensorBoard。当前没有达到这一启动条件。

## 已完成代码及数据工作

1. 增加不跨episode/reset的接触后转移索引。固定batch128配额为接触64、接触前32、接触后16、普通16。未来事件只用于离线采样，未加入模型或Actor当前输入。
2. 任务事件模型支持显式完整lane划分，拒绝重复、交集、越界及遗漏。统一使用物理/旧球模型的96/32划分，从随机参数重新拟合任务模型，未继承旧任务模型或优化器。
3. Tracking参考库排除WM保留lane，启动、resume和extend均检查；冻结采集更新参考库时也应用相同排除集。
4. 新增SQLite索引到参考库转换入口，校验所选archive的数据库manifest、25个原始分块实际SHA256、packed数据及其完成回执、实际任务/资产/评分/执行合同。索引本身不能代替原始轨迹。
5. 配对原生探针只因目标lane终止而拒绝；其他已reset lane不参与最终命令漂移比较。新增足端与身体接触记录。
6. 模式保持、模式切换、未知模式分别报告。模式切换不再作为光滑Jacobian真值，缺少独立边界验证时不签发通过；最低探针支持数同时检查独立episode。

## 两次有固定预算的GPU拟合

只使用RTX5080/cuda:0，顺序运行，无并行GPU训练。原物理、真实奖励、时域、学习率和验收门槛不变。

### 修复采样后的球候选

固定400次更新，batch128、lr2e-4、seed20261117，架构不变。没有扫描参数。

| 锁存边沿相对时刻 | 旧球模型RMSE | 本候选球速RMSE(m/s) |
|---|---:|---:|
| −1 tick | 6.612342 | 5.106146 |
| 当步 | 2.282413 | 1.523254 |
| +1 tick | 0.026637 | 0.273559 |

接触后一步抽样从上一候选15次提高到6,400次，误差相对上一候选1.621674 m/s明显下降；但仍劣于原球模型，**继续拒绝作为Actor学习模型**。接触当步前向符号错误率6.67%→3.33%；后一步为0%。非零模型梯度不构成物理验证。

候选SHA256：`c063c20610644d07ca7e11c8e944a9ba233da38e00168bbe8a65fc8a0c140920`。

### 统一保留集的任务模型

固定512次task optimizer更新，16步窗口、batch256、lr2e-4；GPU进程正常退出。Actor、物理模型和Critic更新均0。

任务模型SHA256：`e66be47371733822e3fc6dc1c22f00e4159fea72f7066bf4a579d44687e8331c`。

修复了划分不一致，**尚未通过任务预测质量验收**。在真实状态输入的一步事件诊断中，合格触球阳性19个、阈值0.5的TP2/FN17；越网14个阳性、TP0；恢复9个阳性、TP0；身体触地16个阳性、TP0。这些是已存在诊断阈值下的事件召回，并非原生任务成绩；总体低Brier不能掩盖稀有事件预测不足。没有将新的模型自动加载到Actor训练中。

## 实际生成成功参考库

通过当前因果执行合同及原始轨迹核验的所选来源：`supermimic_frozen_closure_20261117_223243`。

- Approach参考40条、Recovery参考28条，二者可能属于同一回合，不能相加为独立成功回合数。
- 全部32个保留lane排除；数据库其余历史来源没有因“成功”标签自动纳入。
- 文件：`/home/autodrive/LZQW/experiments/supermimic_admission256_20261118_s5lbit6k/indexed_references/reference_bank.pt`。
- SHA256：`f8dfd70efb05a128cf08bedb24d9135e6d1ec81136b0ff9faac2330b54215534`。
- 此处证明参考数据合同，不证明Tracking改善Actor，也尚未完成实际64步BPTT中的匹配和梯度验收。

## 原生配对探针正在运行

systemd用户服务：`supermimic-native-probe-approach17-20261118-s5lbit6k`，Restart=no，整个组4小时上限，每分支55分钟上限，任何失败停止且不重试。

固定初始化策略SHA8f510fe9f218beef791ac21675a19ed25d11f510a83fe247cddf2893bee08399、seed882072、128环境、headless、GPU PhysX。lane17仅用于诊断，不进入策略规则；272控制tick的请求关节索引12，扰动±0.02，最终命令仍经过原mixer和限速。每个分支运行273tick，不生成正式评分。四分支按nominal_a、nominal_b、minus、plus从地面依次重放。

当前已实际创建128环境，`physics.json`确认device=cuda:0和gpu_dynamics=true，GPU进程出现并占用约3GB。报告生成时nominal_a仍运行，尚无完成或物理响应通过结论。源文件哈希在每个分支启动前复核；不热修改该组执行源码。

这是校准案例，不能冒充独立保留案例；后续仍需不同episode的验证、Recovery探针及多案例汇总。该组不优化任何策略或模型，也不能替代原128×800正式评价。

## 检查及未完成项

实际CPU命令：

```bash
CUDA_VISIBLE_DEVICES='' QUADSMASH_PYTHON=/home/autodrive/Isaac/env_isaaclab/bin/python \
bash scripts/python.sh -m pytest tests/test_supermimic*.py -q -ra --maxfail=1
```

**283 passed，0 failed，0 skipped**，1条既有弃用警告。修改文件语法检查、git diff --check通过。不是GPU BPTT或真实物理验收。

本轮仍缺：合格的动作球/任务模型、双Head原生响应方向验证、当前策略Critic质量、带参数更新的R→A→R、真实resume、长期资源与退化监控验收。没有启动256次Actor长训，也没有以模型拟合日志冒充原生return或新TensorBoard长训记录。
