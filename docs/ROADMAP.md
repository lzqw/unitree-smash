# 阶段状态

| 阶段 | 当前状态 | 证据 |
|---|---|---|
| P0 | 基线已实现 | pyproject、environment.yml、check_env、smoke、文档 |
| P1 | 资产接口就绪，真实 Go2/B2 导入待授权资产 | assets、validate_robot_asset |
| P2 | NumPy 羽毛球飞行与解析碰撞基线已实现 | physics、tests、run_physics_validation |
| P2.5 | 32 环境任务 API 已实现 | QuadSmashEnv、test_env_api |
| P3 | 低层技能 API 已实现，Isaac 场景控制待接入 | control、evaluate_low_level |
| P4 | 事件条件化全身 PPO NumPy baseline 已实现 | event-conditioned observations、WholeBodyPPO、train_ppo |
| P4.5 | 课程、真实接触后恢复、reset bank 和恢复感知事件选择已实现 | goals、events、reset_bank、event_selector |
| P5-P8 | 待 P1-P4 通过量化门槛后推进 | 见原始任务书 |

任何性能提升都要保存固定 seed、配置、checkpoint、事件指标和对应对照实验结果。

