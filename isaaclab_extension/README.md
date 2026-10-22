# Isaac Lab 接入边界

Isaac Lab 场景接入应在这里注册 `QuadSmashEnv` 和 Go2 asset config。当前 Python 基线故意不导入 `omni.isaac.lab`，使物理验收可以在没有 Isaac Sim 的环境运行。接入时必须保留 `get_observations`, `get_rewards`, `get_dones`, `log_episode_metrics` 的字段和维度契约。

