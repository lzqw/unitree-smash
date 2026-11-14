# SuperMimic 原策略任务合同恢复与终止统计修复

2026-11-14。接续 `3397fb1` 的想象对应性审计。本次不更新actor或world model。

## 实际改动

- 新增训练/评估共用的source-task contract：按源schema选择协议；G07必须通过checkpoint内协议、资产、物理配置、案例及精度准入哈希检查，恢复origin0、contact-flight-recovery-v1、源记录的64米地面范围。
- 地面范围来自checkpoint元数据，并核验原 `ground_source_verification.json`。不以新数值替代原记录。
- 训练入口写入完整合同。resume必须包含相同合同；旧16米布局/旧恢复关联下的replay、world、optimizer不能静默续用。
- 环境reset前采集资格触球、`complete5(recovery=all_true)`球侧集合、机器人身体触地和完整违规标志，再与该回合记录绑定。训练保存逐回合原记录，修复将羽毛球落地当机器人身体触地的问题。
- 缺少独立头部/全身接触审计时使用null，不把缺失解释成无接触。
- 重写冻结评估入口，支持直接 `--init-policy-checkpoint` 零更新评估；不创建学习器或optimizer。删除reset后实时状态补写、重复统计代码和`os._exit(0)`；异常单独保存。
- 零更新评估采用历史FrozenPanelEnv的无扰动地面重置过程，并在同一真实状态上每tick独立调用旧完整控制器，与新入口最终18关节目标比较，2e-6容差不变。它验证执行映射，不宣称第二套独立PhysX轨迹完全等价。
- 128个TRAIN案例仍为开发回归；不再将其标记为正式A/B验收。

## 已验证

```text
CUDA_VISIBLE_DEVICES="" QUADSMASH_PYTHON=/home/autodrive/Isaac/env_isaaclab/bin/python \
bash scripts/python.sh -m pytest tests/test_supermimic_task_contract.py \
tests/test_supermimic_horizon_v2.py tests/test_native_action_contract.py -q -ra --maxfail=1

48 passed in 1.61s
```

4个修改模块py_compile通过，git diff --check通过。真实来源policy、协议、物理配置、案例、地面验证及历史精度准入校验通过，见source_task_contract.json。

## 尚未验证及下一步

此提交时尚未完成修正后的原128环境零更新复验，没有新成功率。下一步只运行一次8秒零更新开发评估：覆盖准备、腾空、触球和恢复阶段最终命令，并保留逐回合首回合记录。遇到工程异常或命令不一致即停止，不自动改变容差。

世界模型长时漂移、事件奖励/critic输入、稀有事件回放与物理表示仍未解决；本补丁不宣称已修复这些算法问题。先取得可比的初始化能力，再决定后续物理状态模型与事件学习改进。
