"""Check reference continuity and absence of synthetic support constraints."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.pure_control_settled_stance_v14 import PureControlSettledStanceV14
from scripts.pure_control_stance_foot_reference_v15 import PureControlStanceFootReferenceV15
from scripts.audit_actual_handoff_model_v1 import state_from_record
from scripts.whole_body_support_qp_v1 import PositionPDInterface


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'configuration', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Fresh check output required')
    read = lambda f: json.loads(f.read_text())
    cfg = read(a.configuration)
    nominal = read(a.run/'native_command_inputs.json')['default_joint_position_rad']
    args = (PositionPDInterface(**cfg['interface']), nominal, cfg['model_reference_library'])
    old, new = PureControlSettledStanceV14(*args), PureControlStanceFootReferenceV15(*args)
    with np.load(a.run/'pure_control_commands.npz') as z:
        rows = {k:z[k].copy() for k in z.files}
    diagnostics = [json.loads(line) for line in (a.run/'pure_control_diagnostics.jsonl').read_text().splitlines()]
    selected = None
    for i in np.flatnonzero(rows['lane'] == 50):
        record = {k:rows[k][i] for k in ['q','v','previous','ground_force']}
        record['phase'] = diagnostics[int(i)]['phase']
        s = state_from_record(new.model, record)
        if rows['tick'][i] >= 435 and new.terminal_candidate(s)[0]:
            selected = (int(i), s)
            break
    if selected is None:
        raise ValueError('No actual qualifying terminal state')
    i, state = selected
    snap = new.model.snapshot(state.q, state.v)
    for c in [old,new]:
        c.phase = 'SUSTAIN'
    before = new.reference(state,snap,state.joint_position)
    expected = old.reference(state,snap,state.joint_position)
    if any(not np.array_equal(before[k],expected[k]) for k in expected):
        raise ValueError('Unqualified reference changed')
    for c in [old,new]:
        c.terminal_stance = dict(com=snap['com_world'].copy(),rotation=state.root_rotation_world.copy(),
                                joint=state.joint_position.copy(),captured_tick=0)
    original_contacts = state.contacts.copy()
    new.reference(state,snap,state.joint_position)
    footholds = new.terminal_foot_positions.copy()
    delta = []
    for bits in range(16):
        state.contacts = np.array([bool(bits & (1 << j)) for j in range(4)])
        contacts = state.contacts.copy()
        ref = new.reference(state,snap,state.joint_position)
        base = old.reference(state,snap,state.joint_position)
        if not np.array_equal(state.contacts,contacts):
            raise ValueError('Controller fabricated contact')
        for key in ['com_position','com_acceleration','angular_acceleration_body','joint_acceleration']:
            if not np.array_equal(ref[key],base[key]):
                raise ValueError('Changed unrelated objective '+key)
        if bits:
            if not np.array_equal(ref['swing_foot_target_world'],footholds):
                raise ValueError('Terminal foothold depends on current contact subset')
        elif any(not np.array_equal(ref[k],base[k]) for k in base):
            raise ValueError('No-support reference no longer ballistic')
        delta.append(float(np.max(np.linalg.norm(ref['swing_foot_target_world']-base['swing_foot_target_world'],axis=1))))
    state.contacts = original_contacts
    result = dict(passed=True,actual_case=50,actual_tick=int(rows['tick'][i]),
        contact_subsets_checked=16,maximum_soft_target_difference_m=max(delta),
        physical_constraints_changed=False,native_success=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
