# 直接粘贴进 Codex 的 /goal

先在GPU服务器或可访问该工作区的Codex会话中打开项目。不要在普通shell中执行/goal。
若Codex不识别此命令，按官方文档启用Goal或更新支持该功能的客户端。

## 第一轮：默认只实现与验证G00–G03
```text
/goal 在当前工作区实现 QuadSmash 的 G00–G03。先读取 AGENTS.md、PLAN.md、TRAINING_SPEC.md、ACCEPTANCE.md、configs/budget.json 和 status/STATUS.json，依次执行 goals/G00_environment.md 至 goals/G03_env_ppo.md。直接实现、运行测试并验证，不要只输出计划。只使用已分配的本地GPU，累计最多2 GPU-hours，不修改系统驱动、不创建云资源；没有GPU分配时先完成CPU代码与检查，再报告阻塞。每阶段生成真实证据并运行 tools/verify_goal.py；依赖未通过不进入下一阶段。所有G00–G03通过后停止，或遇到权限/技术阻塞/预算上限时保存状态并停止。不要启动G04训练，也不要把Mock或CPU测试当作Isaac仿真通过。
```

## 第二轮：显式授权第一版训练G04–G07
阅读并认可预算后再使用以下命令；这会授权最多新增60 GPU-hours，不保证达到指标。
```text
/goal 先检查G00–G03真实验收证据。现在授权在既有、已分配的本地GPU上依次执行G04–G07，更新configs/budget.json对应授权：新增累计60 GPU-hours，各阶段8/12/16/24 GPU-hours，同时只运行1个GPU训练进程，不得创建云资源。遵循TRAINING_SPEC.md和对应goals文件，实现并训练完整负载跳跃恢复→空中虚拟击球→真实空中接触恢复→扣杀越网和恢复。所有验收从地面初始化，阈值/球网/质量/力矩限制不得降低。每阶段实测通过tools/verify_goal.py后才继续；最多3轮有记录的修订，达到任一预算或发生阻塞即保存检查点、停止自己的GPU作业并报告。G07通过或明确阻塞后停止，不启动G08。
```

## 第三轮：研究增强G08–G09
```text
/goal 在G00–G07证据有效的前提下，执行G08–G09。授权新增累计28 GPU-hours：G08最多12、G09最多16；只使用已分配本地GPU，最多2个独立实验进程。依次实现物理一致状态库、同预算对照、击球与恢复联合可行性预测及候选事件选择。禁止使用最终测试集调参，禁止把旧rollout当普通PPO on-policy数据，禁止把学习成功概率称为安全证明。报告工程完成状态与研究结论，允许无提升。预算内完成对应goals文件所有证据才标通过，预算不足则保存状态停止，不自动开展G10。
```

## 第四轮：冻结实验G10
```text
/goal 先读取并冻结G10的实验协议、方法列表、5个训练种子和每种子1000回合holdout；禁止读取holdout结果后修改方法。授权新增最多80本地GPU-hours与2个并行实验进程，先运行dry-run预算清单，不创建云资源。按goals/G10_final_study.md完成全部预注册实验和报告，计入预训练、状态库和可行性数据成本。研究结果不必为正，但不得减少种子或修改成功条件来伪装完成。预算不足或技术阻塞时保存可重现部分结果、标未完成并停止。
```

## 单阶段推进（更易定位）
使用下面命令不自动新增预算；阶段必须已在budget.json授权，否则报告BLOCKED_PERMISSION。

### G00 审计环境并锁定兼容软件栈
```text
/goal 读取AGENTS.md、PLAN.md、TRAINING_SPEC.md、ACCEPTANCE.md和goals/G00_environment.md，完成G00，不要推进下一Goal。先验证依赖和预算授权；实现、测试、必要的训练及独立评测，产生真实reports/G00/evidence.json并通过tools/verify_goal.py。达到验收后停止；预算、权限或技术阻塞时保存状态和检查点并停止。禁止降低阈值或以未运行结果声称通过。
```

### G01 组装真实耦合资产并审计执行器
```text
/goal 读取AGENTS.md、PLAN.md、TRAINING_SPEC.md、ACCEPTANCE.md和goals/G01_assets.md，完成G01，不要推进下一Goal。先验证依赖和预算授权；实现、测试、必要的训练及独立评测，产生真实reports/G01/evidence.json并通过tools/verify_goal.py。达到验收后停止；预算、权限或技术阻塞时保存状态和检查点并停止。禁止降低阈值或以未运行结果声称通过。
```

### G02 验证羽毛球飞行、扫掠碰撞与事件
```text
/goal 读取AGENTS.md、PLAN.md、TRAINING_SPEC.md、ACCEPTANCE.md和goals/G02_physics.md，完成G02，不要推进下一Goal。先验证依赖和预算授权；实现、测试、必要的训练及独立评测，产生真实reports/G02/evidence.json并通过tools/verify_goal.py。达到验收后停止；预算、权限或技术阻塞时保存状态和检查点并停止。禁止降低阈值或以未运行结果声称通过。
```

### G03 实现统一任务、训练评测入口与证据链
```text
/goal 读取AGENTS.md、PLAN.md、TRAINING_SPEC.md、ACCEPTANCE.md和goals/G03_env_ppo.md，完成G03，不要推进下一Goal。先验证依赖和预算授权；实现、测试、必要的训练及独立评测，产生真实reports/G03/evidence.json并通过tools/verify_goal.py。达到验收后停止；预算、权限或技术阻塞时保存状态和检查点并停止。禁止降低阈值或以未运行结果声称通过。
```

### G04 训练完整负载下跳跃和落地恢复
```text
/goal 读取AGENTS.md、PLAN.md、TRAINING_SPEC.md、ACCEPTANCE.md和goals/G04_jump.md，完成G04，不要推进下一Goal。先验证依赖和预算授权；实现、测试、必要的训练及独立评测，产生真实reports/G04/evidence.json并通过tools/verify_goal.py。达到验收后停止；预算、权限或技术阻塞时保存状态和检查点并停止。禁止降低阈值或以未运行结果声称通过。
```

### G05 训练按时完成空中虚拟击球事件
```text
/goal 读取AGENTS.md、PLAN.md、TRAINING_SPEC.md、ACCEPTANCE.md和goals/G05_virtual_hit.md，完成G05，不要推进下一Goal。先验证依赖和预算授权；实现、测试、必要的训练及独立评测，产生真实reports/G05/evidence.json并通过tools/verify_goal.py。达到验收后停止；预算、权限或技术阻塞时保存状态和检查点并停止。禁止降低阈值或以未运行结果声称通过。
```

### G06 训练真实来球空中接触并恢复
```text
/goal 读取AGENTS.md、PLAN.md、TRAINING_SPEC.md、ACCEPTANCE.md和goals/G06_real_contact.md，完成G06，不要推进下一Goal。先验证依赖和预算授权；实现、测试、必要的训练及独立评测，产生真实reports/G06/evidence.json并通过tools/verify_goal.py。达到验收后停止；预算、权限或技术阻塞时保存状态和检查点并停止。禁止降低阈值或以未运行结果声称通过。
```

### G07 训练完整扣杀、越网落点和恢复
```text
/goal 读取AGENTS.md、PLAN.md、TRAINING_SPEC.md、ACCEPTANCE.md和goals/G07_smash.md，完成G07，不要推进下一Goal。先验证依赖和预算授权；实现、测试、必要的训练及独立评测，产生真实reports/G07/evidence.json并通过tools/verify_goal.py。达到验收后停止；预算、权限或技术阻塞时保存状态和检查点并停止。禁止降低阈值或以未运行结果声称通过。
```

### G08 研究分支：物理一致中间状态初始化
```text
/goal 读取AGENTS.md、PLAN.md、TRAINING_SPEC.md、ACCEPTANCE.md和goals/G08_reset_bank.md，完成G08，不要推进下一Goal。先验证依赖和预算授权；实现、测试、必要的训练及独立评测，产生真实reports/G08/evidence.json并通过tools/verify_goal.py。达到验收后停止；预算、权限或技术阻塞时保存状态和检查点并停止。禁止降低阈值或以未运行结果声称通过。
```

### G09 研究分支：击球与恢复联合事件选择
```text
/goal 读取AGENTS.md、PLAN.md、TRAINING_SPEC.md、ACCEPTANCE.md和goals/G09_event_selector.md，完成G09，不要推进下一Goal。先验证依赖和预算授权；实现、测试、必要的训练及独立评测，产生真实reports/G09/evidence.json并通过tools/verify_goal.py。达到验收后停止；预算、权限或技术阻塞时保存状态和检查点并停止。禁止降低阈值或以未运行结果声称通过。
```

### G10 冻结评测、多GPU实验与最终报告
```text
/goal 读取AGENTS.md、PLAN.md、TRAINING_SPEC.md、ACCEPTANCE.md和goals/G10_final_study.md，完成G10，不要推进下一Goal。先验证依赖和预算授权；实现、测试、必要的训练及独立评测，产生真实reports/G10/evidence.json并通过tools/verify_goal.py。达到验收后停止；预算、权限或技术阻塞时保存状态和检查点并停止。禁止降低阈值或以未运行结果声称通过。
```

## 会话中断后的继续
```text
/goal 读取status/STATUS.json、status/DECISIONS.md、status/RUNS.jsonl和最后一个Goal报告，先查询已记录PID/Job ID是否仍在运行，禁止重复启动。校验预算余额与checkpoint哈希，在现有授权内继续首个未通过的Goal。已通过且哈希未变化的阶段不重复训练。达到该Goal验收或阻塞/预算边界后保存证据停止。
```

## 查看、暂停、恢复
```text
/goal
/goal pause
/goal resume
/goal clear
```
训练进程另由 scripts/stop_run.py --save-and-stop 或集群调度器管理；先让本项目进程保存
检查点，再暂停会话。不要对共享服务器使用killall/pkill python。
