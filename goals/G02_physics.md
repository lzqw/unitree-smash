# G02：验证羽毛球飞行、扫掠碰撞与事件

前置：G01。证据task_id：`physics_validation`。
累计预算建议：0.5 GPU-hours；所有尝试累计环境交互上限：0（0表示本阶段无正式学习预算）。

先在独立CPU单元测试实现重力+二次阻力、运动/旋转拍面扫掠碰撞和冲量响应，再对接GPU。
定义不少于50个解析接触/不接触/擦边/旋转/离开后再接触案例。对无阻力轨迹与解析解
比较；无重力有阻力时检查相对空气速度衰减；旋转拍面验证接触点速度。
禁止重复冲量；自定义模式处理等量反作用并与引擎接触互斥。球网和落地也做扫掠事件。
用dt=2/1/0.5ms、同policy_dt做收敛性检查，明确近边界不稳定案例。
非边界接触基准建议出球速度差≤5%、方向差≤5°、接触时刻差≤2ms；解析单元测试使用
更紧容差并公开。不能靠改球拍碰撞体让测试通过。

`python scripts/validate_physics.py --num-envs 32 --output reports/G02`

超高速穿透时先修检测/时间步，不用更大的隐形拍面；反作用API不可用则修接口或阻塞。

## 机器验收指标

| 指标 | 要求 |
|---|---|
| `analytic_flight_ok` | `eq True` |
| `swept_contact_ok` | `eq True` |
| `no_double_impulse` | `eq True` |
| `reaction_impulse_ok` | `eq True` |
| `timestep_convergence_ok` | `eq True` |

## 必须交付

- `reports/G02/REPORT.md`
- `reports/G02/commands.log`
- `reports/G02/physics_tests.json`

另外生成 `reports/G02/evidence.json`，所有指标来自实际程序输出，
运行 `python tools/verify_goal.py --root . --goal G02`；通过后才更新STATUS为PASSED。
