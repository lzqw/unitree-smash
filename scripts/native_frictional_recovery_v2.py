"""Native measured-state adapter with bounded local restoration before CEM."""
import atexit
import json
from scripts.native_frictional_recovery_v1 import NativeFrictionalRecoveryV1
from scripts.frictional_receding_recovery_v2 import FrictionalRecedingRecoveryV2


class NativeFrictionalRecoveryV2(NativeFrictionalRecoveryV1):
    def __init__(self,bundle,case,limits,policy_sha256,output,warm_plan,sphere_audit):
        super().__init__(bundle,case,limits,policy_sha256,output,warm_plan,sphere_audit)
        self.planner.close()
        from pathlib import Path
        spheres=json.loads(Path(sphere_audit).read_text())['spheres']
        self.planner=FrictionalRecedingRecoveryV2(self.manifest['configuration'],spheres,horizon=80)
        atexit.register(self.planner.close)

    def save(self):
        result=super().save()
        result['planner_version']='local_restoration_v2'
        path=self.output/'sequence_native_result.json';tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(path)
        return result
