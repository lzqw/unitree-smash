# 便携CPU重放

需要三份已校验的数据：本目录报告所链的6.77MB恢复参考包、1.71MB六例公开输入NPZ，以及64.12MB运行时资产配套包。使用配套包的 `restore_inputs.py` 在 `/tmp` 前缀内恢复原路径记录；固定Python/Torch环境仍从外部复用。这里重放控制命令，没有重跑Native物理。

解压6.77MB包后，可直接使用其 `delivery/replay_fresh.py`，参数如下。`--runtime-root` 指恢复树中的 `home/autodrive/LZQW/unitree_smash_contact_recovery`，`--package-root` 指6.77MB包的解压目录，输出使用尚不存在的临时JSON文件。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /path/to/pinned/python /tmp/packet/delivery/replay_fresh.py \
  --runtime-root /tmp/restored/home/autodrive/LZQW/unitree_smash_contact_recovery \
  --package-root /tmp/packet \
  --output /tmp/recovered_trace_cpu_check.json
```

该入口重放真实800行恢复实录。全部9个候选的六例重放使用本仓库的 `replay_family_public.py`；先把它复制到解压包的临时 `delivery` 目录，再运行：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /path/to/pinned/python /tmp/packet/delivery/replay_family_public.py \
  --runtime-root /tmp/restored/home/autodrive/LZQW/unitree_smash_contact_recovery \
  --package-root /tmp/packet \
  --public-data /tmp/G07_measured_swing_reference_actual_public_actions_6cases_20261027.npz \
  --output /tmp/all9_public_cpu_check.json
```

检查器从恢复根目录运行，读取恢复后的步行权重与URDF，拒绝Python读取原现场项目。实际执行的两份回执记录最大误差分别1.001358e−5和原样对照6.556511e−6，均保留原5e−4容差；全部9项每项4800行已执行。候选权重容器保持 `native_execution=false`，这些CPU结果不产生物理成功、恢复率或正式面板成绩。
