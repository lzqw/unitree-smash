from pathlib import Path
from datetime import datetime, timezone
import hashlib,json
from pxr import Usd,UsdPhysics
root=Path(__file__).resolve().parents[3]
p=root/'assets/quadsmash_standard_v2.usd'
s=Usd.Stage.Open(str(p),load=Usd.Stage.LoadNone)
for prim in s.Traverse():
    if not prim.IsA(UsdPhysics.FixedJoint):continue
    joint=UsdPhysics.FixedJoint(prim)
    if not any(str(x).endswith('/Racket') for x in joint.GetBody1Rel().GetTargets()):continue
    q=joint.GetLocalRot0Attr().Get();q1=joint.GetLocalRot1Attr().Get()
    report=dict(verified_at=datetime.now(timezone.utc).isoformat(),asset=str(p),asset_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),joint=str(prim.GetPath()),body0=str(joint.GetBody0Rel().GetTargets()[0]),body1=str(joint.GetBody1Rel().GetTargets()[0]),local_position0_m=list(joint.GetLocalPos0Attr().Get()),local_position1_m=list(joint.GetLocalPos1Attr().Get()),local_rotation0_wxyz=[q.GetReal(),*q.GetImaginary()],local_rotation1_wxyz=[q1.GetReal(),*q1.GetImaginary()])
    assert abs(report['local_position0_m'][0]-.18)<1e-7 and report['local_rotation0_wxyz']==[1,0,0,0]
    Path(__file__).with_name('LEGACY_MOUNT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
    break
else:raise AssertionError('Legacy physical racket fixed joint missing')
