# Event-latch world-model alignment patch

日期：2026-11-14

## 目的

长训练审计发现，原生奖励使用一次性事件锁存位，而794维actor观测不包含这些锁存位。world model 因而无法严格区分“事件尚未发放”和“事件已经发放”的相同物理观测。本补丁只扩展 world-model/replay 的辅助状态，不改变 actor 的794维输入、18维 residual 动作接口、Isaac物理、正式判定或原生奖励。

## 实现

- `SuperMimicEnvV1`暴露8维事件锁存状态：`jump/contact/cross/target/recovery/body/frame/fault`。
- reset 前保存终止回合的最终事件状态，避免自动 reset 后丢失标签。
- `SequenceReplay`保存`event_state`和`next_event_state`，并检查续训布局一致性。
- `SuperMimicWorldModel(event_dim=8)`以事件状态作为输入，预测下一事件锁存logits。
- imagined rollout使用单调锁存更新：`e_next=e+(1-e)*sigmoid(logit)`。
- world-model监督增加下一事件状态的BCE项；teacher-forced和autoregressive分支都保持事件状态。
- 旧无事件 replay/checkpoint 不会被静默加载到新布局。

## CPU验证

```text
38 passed in 1.82s
py_compile passed
```

另完成一个独立CPU event-aware trainer smoke：world update、imagined rollout、actor/critic更新均成功，事件状态维度和梯度路径正常。

## 限制

这只是语义对齐修复，还没有新的Isaac训练成绩。需要用新鲜原生转移启动新训练；旧长训练checkpoint/replay不兼容该事件状态布局。下一轮应同时记录world-model单步/16步事件预测误差，并与正式冻结评估分开。
