# 状态依赖与接入断点

| 状态或计算 | 真实用途 | 既有记录 | 本轮处理/尚缺工作 |
|---|---|---|---|
| 55维机器人与球状态 | 物理模型输入 | 有，命名关节顺序已保存 | 保留，不称完整Markov状态 |
| 11×69历史、goal、clock、最终归一化动作 | 重建actor的794维输入 | 原始archive有 | 已有独立重建工具；新的完整想象闭环尚未验证 |
| G07嵌套基底、规划参数、event_time、target_offset、valid_plan、速率历史 | 当前状态产生base，再与actor残差混合 | tensors/values有大部分，aliases20项，模块权重单独加载 | 完整复制、逐lane reset、推进与数值同一性尚未通过 |
| reference.logs与plan_cursor | SharedTargetController等读取未消费计划记录，更新valid_plan | logs被误标为diagnostic而省略 | 新采集保留真实FullRacketRallyControllerV4.logs；旧记录不补空列表。可进一步核查原始规划输出或从reset重放补齐，未宣称不可恢复 |
| nn.Module与bound method | 历史源actor、关节准备、运动学及被重绑定的teacher.action | omitted里标network/code | 权重/源码可哈希定位；不能把所有模块直接当无状态或在branch间共享可变对象 |
| contact_recovery稳定窗口、飞行绑定、着地资格 | 当前恢复状态及下步奖励 | 只存结果标志，没有stable_run、flight_id等完整记忆 | 真实阻塞：新采集增加全部原类tensor字段。10ms边界不能无证据恢复1ms内稳定中断 |
| jump_metrics累计高度、连续净空、takeoff/landing等 | 合格腾空和恢复判定 | 仅若干汇总flag | 新采集加入原FLOAT_FIELDS/BOOL_FIELDS，未改判定 |
| _observed_air_time与足地滞回 | 20ms持续腾空判定 | 足端标志有，累计airtime无 | 新采集显式保存。仅从一帧foot flag不足以重算 |
| ContactEpisodes.active/count/clear_frames | 接触去重和分离后的再武装 | rally计数有，独立接触计数器未完整保存 | 新采集保存；没有子步接触序列时不可伪造 |
| rally计时、quiet_frames与首碰/分离/落点等 | 一次性任务事件 | 原archive task包含40余字段 | 可直接复用；新模型如何因果更新全部字段尚未实现 |
| applied_torque、根COM速度、gravity、safety heights、action_delta | 原生连续奖励 | 部分已存或可由资产/运动学推导，新reward_inputs布局整体不存在 | 新采集直接保存精确输入；并非所有旧信息都丢失。旧数据应逐项追溯映射，完整奖励对齐未运行 |
| command_limits | 最终目标解码及限幅 | 可从固定资产/原执行配置定位，但旧archive未直接存 | 新采集直接保存，禁止自行放大。未把这一缺项等同物理数据不可恢复 |
| critic事件/时间/执行上下文 | 匹配state的bootstrap和GAE | 当前critic只接收obs794；奖励事件8未作为critic额外输入 | 本轮未推进到critic重构/训练 |
| PhysX接触缓存 | 精确原生中途状态恢复 | 不可直接视为已捕获 | 明确not_captured；未来动作敏感性使用同前缀重放，先检验重复性 |

## 现有学习调用路径（静态源码证据）

- `scripts/train_supermimic_horizon_v2.py:225`仍构造`SuperMimicWorldModel(...structured_history=True,event_dim=8)`。
- `quadsmash/rl/supermimic_horizon_learner_v2.py:145`调用`self.world(obs, action, event_state)`，随后`obs+delta`。
- 同文件165行critic仍是`self.critic(obs)`。
- 拍头候选和原物理模型均是`PhysicalWorldV2`、schema `physical55-separated-contact-transitions-v2`；本轮合同审计没有调用其forward，调用数0。
- `FullRacketRallyControllerV4.action`仍含`@torch.inference_mode()`以及`.cpu().numpy()`规划；SharedTargetController.action也在inference_mode下。已有外层mixer可导不证明完整基底连续计算可导。后续需分别处理连续计算、离散选支和已声明的stop-gradient，不能以一次residual梯度非零替代完整链。

## 审计计数解释

`archive_audit/result.json`的missing_auxiliary_state表示新合同字段组未出现在旧记录中，不表示组内每个量都无法追溯。真正足以停止的是缺失的连续稳定/触球飞行关联等因果记忆，且原archive没有1ms级完整稳定/接触事件轨迹。没有填零，也没有修改旧文件。
