# G00：审计环境并锁定兼容软件栈

前置：无。证据task_id：`environment_audit`。
累计预算建议：0.5 GPU-hours；所有尝试累计环境交互上限：0（0表示本阶段无正式学习预算）。

检查当前工作区、Git改动、GPU分配、驱动、Python、Isaac Sim/Lab、RSL-RL与资产路径。
不要自动占用他人的GPU，不先执行远程一键安装。选择兼容版本并锁定SHA与完整依赖。
优先运行锁定版本的官方简单任务：32环境、至少200策略步、PPO至少2次更新、保存和
重载策略。建立项目包及scripts/audit.py，登记代码和依赖许可。禁止用FakeEnv通过。
本阶段checkpoint只是训练链路冒烟，不代表机器狗学会任何动作。

命令契约：`python scripts/audit.py --output reports/G00/environment.lock.json`
再运行对应官方任务，将原始命令、日志和检查点重载结果写入证据。

无GPU、无分配权限、缺Isaac许可证/依赖或导入失败时BLOCKED，并给可复现错误。
若参考仓库依赖不兼容，迁移少量资产/配置或选择兼容提交，不循环升级整个服务器。

## 机器验收指标

| 指标 | 要求 |
|---|---|
| `gpu_smoke_ok` | `eq True` |
| `dependency_lock_ok` | `eq True` |
| `checkpoint_reload_ok` | `eq True` |

## 必须交付

- `reports/G00/REPORT.md`
- `reports/G00/commands.log`
- `reports/G00/environment.lock.json`

另外生成 `reports/G00/evidence.json`，所有指标来自实际程序输出，
运行 `python tools/verify_goal.py --root . --goal G00`；通过后才更新STATUS为PASSED。
