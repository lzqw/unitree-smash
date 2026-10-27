# 已完成九组搜索与独立128例交付

本包包含九项实际权重、全部1152个原始搜索回合、选中0的独立128回合、全128机身/球50Hz记录、六个诊断lane的100Hz公共输入与指令、真实地面初始化、接触事件、源快照、作业回执与计分配置。九项均为固定原actor上的教师参数搜索，不是新actor训练。正式A/B与G07性能目标未通过。

`MANIFEST.json`逐项记录字节数、SHA及来源。内容相同的文件仅存一份；`aliases`是逻辑原名到实际包内文件的映射，复现程序自动解析，不应将别名缺少单独实体误判为缺文件。

运行时配套文件仍需同一Release中的 `G07_contact_recovery_v5_cold128_complete_inputs_and_assets_20261027.zip`，64,120,557字节，SHA256 `e0a9e87cdec749caf39d4f86230e7851e35d7dddc13ec8af425809fee798db1d`。它包含原Go2步行策略、标准PiPER资料和完整USD依赖。按该包说明恢复到隔离目录，本包复算程序会核对实际运行的394份源码。请使用既有固定Python/Torch/Isaac环境，不假定仅安装此仓库即可运行原生仿真。

解压本包后，可执行：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /path/to/pinned/python delivery/replay_downloaded.py \
  --packet /path/to/unpacked-packet \
  --runtime /path/to/restored/unitree_smash_contact_recovery \
  --output /tmp/downloaded-swing-replay.json
```

该程序从下载的权重、实际控制器工厂和对应真实公共输入重放全部九组及独立128评估的六例指令，并独立复算1280回合计分；没有读取本机原项目Python的后门。它不会再次执行物理。原生重跑需要GPU0、固定Isaac环境、完整资产路径与现有单作业guard/账本；物理步骤和原成功判定保持，不能把CPU指令一致称为新原生成绩。

实际执行源码是每个运行的 `execution_source_snapshot.zip`/`execution_source_hashes.json`；guard同时保留的Main `source_snapshot.zip`是另一个范围，不应将两者混同。未来提前横向、短窗口、恢复或腿部反向调姿的结果均不包含在本包中。
