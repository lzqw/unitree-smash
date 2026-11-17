"""Same-input default QP regression against preserved pre-change source."""
import importlib.util,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path('tests').resolve()))
from test_admitted_arm_support_qp_v1 import fixture
from scripts.implicit_command_support_qp_v1 import ImplicitCommandSupportQP
from scripts.whole_body_support_qp_v1 import SupportSettings
path=Path(__file__).parent
spec=importlib.util.spec_from_file_location('previous_implicit_qp',path/'PREVIOUS_IMPLICIT_QP.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
rows=[]
for arm in (True,False):
    _,interface,state=fixture();state.previous_command_target=state.joint_position.copy()
    plans=[cls(interface,SupportSettings(),preserve_arm_command=arm,arm_target_rate_rad_s=10.).solve(state,state.joint_position)
        for cls in (module.ImplicitCommandSupportQP,ImplicitCommandSupportQP)]
    assert plans[0]['accepted']==plans[1]['accepted']
    errors={k:float(np.max(abs(plans[0][k]-plans[1][k]))) for k in
        ('acceleration','torque','force_world','joint_position_target','predicted_joint_position','predicted_joint_velocity')}
    assert all(v==0 for v in errors.values()),errors
    rows.append(dict(preserve_arm=arm,accepted=plans[1]['accepted'],max_abs_differences=errors))
(path/'DEFAULT_REGRESSION.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows))
