# 环境记录

本机检查日期：2026-10-21。

| 项目 | 当前记录 |
|---|---|
| GPU | NVIDIA GeForce RTX 5080 |
| 驱动 | 595.91.07 |
| 显存 | 16303 MiB |
| 系统 Python | 3.12.3 |
| Isaac Lab 源码 | `/home/autodrive/Isaac/IsaacLab` |
| Isaac Lab 主线 Python | 3.11（由其 environment.yml 记录） |
| NumPy/Torch/Gymnasium（系统 Python） | 未安装；请在项目环境或 Isaac Lab 环境安装 |

每次正式训练须把 `python --version`、`pip freeze`、Isaac Sim / Isaac Lab commit、CUDA 与驱动信息写入 `logs/`。`scripts/check_env.sh` 会输出基础硬件和模块探测结果。

