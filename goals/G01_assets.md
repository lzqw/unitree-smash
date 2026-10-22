# G01：组装真实耦合资产并审计执行器

前置：G00。证据task_id：`asset_audit`。
累计预算建议：0.5 GPU-hours；所有尝试累计环境交互上限：0（0表示本阶段无正式学习预算）。

加载Go2+标准PiPER+固定工具；核对18个主动关节，所有刚体质量/质心/惯量、关节限制、
工具安装位置和实际球拍接触点。夹爪处理必须与后续资产一致。组合为一个正确耦合的
浮动基座系统；记录模型来源与哈希，逐项区分真实参数和假设参数。
运行100次、每次10s站立及低速摆臂；至少95%站立成功。对腿/臂分别做小幅关节映射测试。
无地面接触的诊断中，确认机械臂动作会引起机身反应，并检查总质心/角动量数值一致性；
不把姿态传送当作动力学连接。标定足接触阈值、碰撞几何和关节限幅。
能力审计先覆盖可达空间、实际拍点速度、负载与静态力矩；动态起跳能力由G04验证。

`python scripts/validate_assets.py --num-envs 32 --episodes 100 --output reports/G01`

无法核实安装板/球拍参数时允许写入显式估计继续仿真，但必须标注未实机校准。
若官方模型缺质量/惯量，不得填零或无限小值，报告并请求合法参数/建模决策。

## 机器验收指标

| 指标 | 要求 |
|---|---|
| `joint_mapping_ok` | `eq True` |
| `coupled_dynamics_ok` | `eq True` |
| `asset_manifest_ok` | `eq True` |
| `stand_success_rate` | `ge 0.95` |
| `sample_count` | `ge 100` |
| `stand_episode_seconds` | `ge 10.0` |

## 必须交付

- `reports/G01/REPORT.md`
- `reports/G01/commands.log`
- `reports/G01/assets_manifest.json`
- `reports/G01/asset_tests.json`

另外生成 `reports/G01/evidence.json`，所有指标来自实际程序输出，
运行 `python tools/verify_goal.py --root . --goal G01`；通过后才更新STATUS为PASSED。
