# SuperMimic 原生准入三环节代码实现

提交：`dc79dfbb3612881b919002323e087a41a485fdaa`
基于：`1af5c1fbbe2a48711c6fb0ce7a7754999c88c8bd`

## 已实现

### 候选安全筛选

保留并验证 `supermimic_wm_candidate_gate_v1.py` 与
`scripts/promote_supermimic_wm_candidate.py`。候选必须在固定留出窗口上具备完整指标，接触边缘球速指标需要改善，接触前和接触后相邻指标不得退化。缺失、非有限、退化或没有接触边缘改善时生成拒绝决策，不复制候选 checkpoint。

### PhysX 事件对齐探针

已有链路现在可被固定命令执行：

1. `build_supermimic_event_aligned_probe_specs.py` 从留出 lane 和真实事件边缘生成规格。
2. `run_supermimic_event_aligned_native_probes.py` 对每个规格启动 `nominal_a`、`nominal_b`、`minus`、`plus` 四个独立 headless GPU PhysX 子进程。
3. `build_supermimic_probe_region.py` 只接受同前缀、同执行器记忆、同任务合同的四分支真实产物。
4. `aggregate_supermimic_native_probes.py` 只聚合独立单探针 bundle，不接受重复字节冒充独立支持。

探针 runner 在缺少 CUDA、分支失败、缺少 `native_probe.pt` 或输出目录已存在时退出，不写完成标记，不合成 PhysX 数据。

### 原生动作响应方向验证

新增 `scripts/validate_supermimic_native_action_response.py`。它要求：

- 真实 `native-paired-prefix-region-v1` bundle；
- 预先声明且有限的事件误差、响应 RMSE、方向误差、幅值误差和最小样本数；
- 训练 episode 与验证 episode 完全不相交；
- 当前 Actor、任务合同、执行代码和策略身份与探针兼容；
- native nominal 重复性、最终命令扰动、接触模式、原生响应与 WM 响应方向、局部 Jacobian 投影均通过。

只有全部通过才写入 `supermimic-contact-validation-v1` receipt，并立即重新调用 `verify_contact_receipt` 校验模型哈希、合同哈希和证据文件哈希。失败时只写诊断结果，不生成 receipt。

`verify_contact_receipt` 现在还要求 `response_direction_verified=true` 和独立验证 episode 支持，不能只凭 paired sample 数量通过。

## 代码路径修复

隔离 worktree 下的 G07 执行器现在通过 `QUADSMASH_G07_EXECUTION_ROOT` 统一解析 PiPER URDF、球拍配置、G07 配置和旧策略相对路径，避免代码来自临时 worktree 而资产来自错误根目录。

## 测试

解释器：`/home/autodrive/Isaac/env_isaaclab/bin/python`

命令：

```bash
QUADSMASH_PYTHON=/home/autodrive/Isaac/env_isaaclab/bin/python \
  bash scripts/python.sh -m pytest -q -ra --maxfail=1 tests/test_supermimic_*.py
```

结果：`425 passed, 1 warning`。

定向准入测试：`18 passed`。
脚本和修改模块 `py_compile` 通过。

## 当前未完成的物理步骤

本轮没有启动 Isaac、原生探针或训练。服务器当前 `nvidia-smi` 无法连接驱动，`torch.cuda.is_available()` 为 `False`，设备数为 `0`。因此：

- 候选退化拒绝逻辑已通过 CPU 合同测试。
- 事件对齐探针的代码链已实现，但没有真实 PhysX 分支产物。
- 原生动作响应方向 validator 已实现，但没有真实 bundle，因此没有生成 contact-validation receipt。
- 没有声称 WM 方向准确，也没有开放 Contact-aware Actor BPTT 或大规模训练。

GPU 恢复后，必须按规格生成、四分支原生采集、区域构建、独立事件误差和方向 validator 的顺序执行。只有 receipt 通过，才能进入 R8→A8→R8 小规模闭环，再由用户确认是否启动大规模训练。
