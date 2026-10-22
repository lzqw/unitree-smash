# G00 environment audit

The project is now running in the dedicated `/home/autodrive/Isaac/env_isaaclab` environment: Isaac Lab 2.3.0 source, Isaac Sim 5.1.0.0, PyTorch 2.7.0+cu128, RSL-RL 3.0.1, and Python 3.11.17. `nvidia-smi` and PyTorch both see the RTX 5080 with CUDA 12.8; the run declared `CUDA_VISIBLE_DEVICES=0`.

A real Isaac Lab/PhysX Go2 smoke completed with finite tensors. A short native RSL-RL PPO run wrote `reports/G00/native_train/checkpoint.pt`, and a separate evaluation process loaded that checkpoint successfully. The reduced NumPy smoke remains development evidence only.

The composed USD was rebuilt with an explicit root-layer override for the PiPER root joint; the refreshed environment lock records the new coupled asset hash.

The final coupled asset also contains the parameterized standard_racket_v1 rigid payload and RacketMount.
