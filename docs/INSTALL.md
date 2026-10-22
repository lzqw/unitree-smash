# QuadSmash-Sim 安装

工程按阶段运行。当前 P0-P2.5 的物理和任务接口只需要 Python 3.10+ 与 NumPy；Isaac Sim / Isaac Lab 是 P1 场景接入和后续全身训练的可选依赖。

```bash
cd /home/autodrive/LZQW/unitree_smash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest
bash scripts/run_smoke_test.sh
```

Isaac Lab 主线 checkout 在本机 `/home/autodrive/Isaac/IsaacLab`。请按照该 checkout 的官方安装说明创建其 Python 3.11 环境，再执行本项目的 `pip install -e .`。不要为了复用旧任务随意修改 Isaac Lab 源码或降级一组依赖。

`checkpoints/`、`logs/` 和运行报告由脚本生成，不应提交模型二进制或未授权的 Unitree 资产。

