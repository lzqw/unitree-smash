import json
from pathlib import Path
import numpy as np
import pytest
from scripts.compare_native_model_properties_v1 import compare
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
from scripts.whole_body_support_qp_v1 import PositionPDInterface


def test_realized_properties_match_and_perturbation_is_not_hidden():
    root=Path('reports/prediction_error_attribution_20261118/property_inspection01/native')
    metadata=json.loads((root/'PROPERTIES.json').read_text())
    with np.load(root/'PROPERTIES.npz') as z:arrays={k:z[k].copy() for k in z.files}
    model=LoadedPinocchioModelV1()
    interface=PositionPDInterface(**json.loads(Path('reports/codex_teacher_joint_eval_20261116/controller.json').read_text())['interface'])
    result=compare(model,interface,metadata,arrays)
    assert max(b['inertia_max_delta_kg_m2'] for b in result['bodies'])<1e-7
    assert max(result['interface_max_differences'].values())<1e-6
    assert result['armature_max']==0 and result['friction_max']==0
    arrays['masses'][0,0]+=.25
    altered=compare(model,interface,metadata,arrays)
    assert max(b['mass_max_delta_kg'] for b in altered['bodies'])>=.249
    assert not altered['complete_dynamics_equivalence']
    arrays['masses'][0,0]=np.nan
    with pytest.raises(ValueError,match='Nonfinite'):compare(model,interface,metadata,arrays)
