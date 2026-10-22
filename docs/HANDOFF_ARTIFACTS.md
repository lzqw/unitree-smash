# GitHub 交接附件与恢复说明

仓库：`lzqw/unitree-smash`（私有）。交接标签：`handoff-20261022`。
[下载此次附件](https://github.com/lzqw/unitree-smash/releases/tag/handoff-20261022)。

Git 主分支保留代码、配置、任务规范、状态、原始 JSON/CSV、文本日志、源码快照和图表。Release 附件保存以下现有文件的原始字节，不改写已经出具的证据：

| 附件 | 内容 |
|---|---|
| `evidence.tar.gz` | 完整 `reports/`，包括原生验证使用的模型文件和历史失效结果 |
| `checkpoints-and-training-records.tar.gz` | `artifacts/`、全部 `runs/` 非 `.pt` 记录，以及修正后各 G04 训练作业的最终 `native/checkpoint.pt` |
| `runtime-assets.tar.gz` | 项目资产、标准 PiPER 实际使用的 URDF/网格及转换 USD、外部 Go2 USD 文件 |
| `handoff_manifest.json` | 逐文件和逐附件大小、SHA256、路径映射、排除范围 |
| `SHA256SUMS` | 下载后验证附件的摘要列表 |

约 18 GB 的每次更新 `model_*.pt`、完整第三方仓库、Python/Isaac 安装与缓存不上传，仍保留在原机器。保留了最新可恢复检查点及待讨论方案所需的初始检查点。附件不是整个服务器备份。

## 仅供阅读和讨论

先看 [GPT 交接说明](GPT_HANDOFF_20261022.md)，无需下载模型或运行 GPU。私有仓库需要使用有权限的 GitHub 登录或相应连接器读取；直接把 URL 交给未授权的阅读工具可能读不到。也可将这份说明和所列核心代码提供给讨论方。

## 下载与离线证据验证

在一个新的 clone 内操作，以免覆盖现有结果：

```bash
gh repo clone lzqw/unitree-smash
cd unitree-smash
gh release download handoff-20261022 --repo lzqw/unitree-smash --dir .handoff_export/download
cd .handoff_export/download
sha256sum -c SHA256SUMS
cd ../..
tar -xzf .handoff_export/download/evidence.tar.gz
tar -xzf .handoff_export/download/checkpoints-and-training-records.tar.gz
tar -xzf .handoff_export/download/runtime-assets.tar.gz
python3 tools/verify_goal.py --root . --goal G04
```

最后一步预期 **exit 2 / G04 FAILED**：必须如实保留 0 < 0.90 的性能失败；G00–G03 依赖、文件及摘要应通过。只 clone 代码、未解压附件时，会因缺少二进制证据而额外失败。

## 原生运行环境与已知移机限制

实际主线为 Python 3.11.17、Isaac Sim 5.1.0.0、Isaac Lab 源码 2.3.0（Python 包元数据 0.47.2）、PyTorch 2.7.0+cu128、RSL-RL 3.0.1、Gymnasium 1.2.0、NumPy 1.26.0、RTX 5080 / 驱动 595.91.07。详见 `reports/G00/environment.lock.json`；其中旧模型字段是 G00 当时的记录，当前标准资产以 G01 最新证据为准。

这是现有研究工作区的交接，**尚未改造成任意目录一键运行的便携项目**：

- 组合 USD 内保留原始绝对引用：`/home/autodrive/Isaac/unitree_model/Go2/usd/go2.usd` 与 `/home/autodrive/LZQW/unitree_smash/assets/robots/piper_standard/piper_standard.usd`。
- 附件中的 Go2 位于 `external/Go2/`；在新机器上需显式安排同样的路径映射，或经过审查后重建资产引用。重建改变 USD 哈希时，严格检查点兼容校验会拒绝加载，不应直接删掉该检查。
- 原机器解释器为 `/home/autodrive/Isaac/env_isaaclab/bin/python`，`scripts/python.sh` 支持 `QUADSMASH_PYTHON` 覆盖；部分原生入口仍有原机器路径。历史日志/manifest 的绝对路径只记录来源，不会自动重定位。
- 附件没有第三方完整 `.git` 历史。若需重新 URDF 转换，按 `assets/standard_piper_manifest.json` 的来源和 commit 恢复厂商 checkout；保留厂商文件自身声明，不将第三方资产视为本项目新授权。

本次没有重新启动 Isaac/GPU 仿真来验证移机运行。CPU 证据核验与附件完整性检查的范围应与实际运行能力区分。

原机器可用的评测入口如下；这是**参考命令，不是新增执行授权**，新评测也要计入累计预算，输出目录必须全新：

```bash
python3 scripts/evaluate.py --stage jump_recover \
  --checkpoint artifacts/G04/policy.pt --config configs/jump_ppo_v2c.json \
  --suite dev_jump_v2 --episodes 400 --num-envs 200 --reset-mode ground \
  --output reports/G04/NEW_EVALUATION_DIRECTORY
```

最新评测权重 SHA256：`0c94cecf197b9dd5ab658fe391e42122fbbefdc8e10227d90c4f05f8ab26818f`。
待讨论低噪声方案的初始权重为 `runs/G04/corrected_v2c_main01/native/checkpoint.pt`，SHA256：`ebda7c124da703fc0b5be99306e2c33feb15db1fac4f455c354277f91c02ad51`。

用户此次请求是上传交接，额外训练修订仍未获授权；G04 总目标保持未完成。
