import json
from pathlib import Path
p=Path(__file__).parent/'REPORT.json';r=json.loads(p.read_text())
assert r['completed_generations']==4 and r['actual_control_env_steps']==409600
assert r['counts']['complete_smash']==0 and r['counts']['numerical_failures']==0
assert r['native_execution'] and not r['formal_acceptance']
print('CEM24 terminal evidence verified: 4 complete generations, 409600 steps, full=0, NF0.')
