# evidence.json 契约

文件路径：reports/Gxx/evidence.json。由真实评测程序写入，不手工编造结果。
`contract_sha256`是configs/acceptance.json的SHA256。改协议将使旧证据失效，须重新审计。

```json
{
  "schema_version": 1,
  "goal_id": "G04",
  "task_id": "jump_recover_ground_v1",
  "contract_sha256": "实际验收配置文件SHA256",
  "evidence_kind": "observed",
  "run": {
    "command": "实际执行命令",
    "git_commit": "真实commit；dirty差异另存",
    "started_at": "实际ISO时间",
    "finished_at": "实际ISO时间",
    "gpu_hours": 0.0,
    "env_steps": 0
  },
  "metrics": {},
  "artifacts": [
    {"path": "reports/G04/REPORT.md", "sha256": "真实文件SHA256"}
  ]
}
```
示例故意没有成功指标和有效哈希，不能用于通过验收。所有required_artifacts都要列出。
每个artifact必须在工作区内，非空，哈希一致。raw CSV、配置和checkpoint还要在报告里
建立对应关系，指标必须能由CSV重算。工具不是远程真实性证明；SHA只证明文件未变。

验证命令：
```bash
python tools/verify_goal.py --root . --goal G04
```
只做结构/预算之外的门槛核验：预算实际限制由训练runner实施；不能只靠此工具控制GPU。
