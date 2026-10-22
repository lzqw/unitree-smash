# G04 通过后的下载与恢复

代码与文字证据分支：`g04-guided-learning-20261022`。

权重与大文件：[g04-passed-20261022 release](https://github.com/lzqw/unitree-smash/releases/tag/g04-passed-20261022)。下载 `g04-passed-policy-and-trace.tar.gz` 和 `SHA256SUMS`，校验后在仓库根目录解压：

```bash
sha256sum -c SHA256SUMS
tar -xzf g04-passed-policy-and-trace.tar.gz
```

包中包含最佳原生策略、TorchScript actor/接口、两轮 BC 的初始化检查点和完整配对诊断轨迹。逐文件哈希见 `release_manifest.json`。运行资产及旧环境交接仍使用之前的 `handoff-20261022` release。完整 BC/DAgger 观测数据保留在本机 runs 目录，未纳入此下载包；可按记录的种子、教师参数、命令和对应源码快照重新生成。
