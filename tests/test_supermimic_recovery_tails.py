"""CPU archive fixture for unperturbed Recovery data extraction."""
import json

import pytest
import torch

from quadsmash.rl.supermimic_physical_capture_v1 import PhysicalTransitionArchive
from quadsmash.rl.supermimic_terminal_scoring import APPROACH_BALL_SCORING
from scripts.export_supermimic_recovery_tails import MIXER_STATE, STATE_FIELDS, export_run
from scripts.index_supermimic_recovery_tails import index_run


def test_frozen_native_capture_exports_only_non_holdout_lanes(tmp_path):
    lanes = 128
    task_contract = {'approach_ball_scoring': {'scoring_schema': APPROACH_BALL_SCORING}}
    execution = {'arm_preparation_seconds': .25}
    metadata = dict(task_contract=task_contract, policy_execution=execution, actor_state_sha256='actor',
                    policy_id='frozen-evaluation', wm_holdout_lanes=[1], num_envs=lanes,
                    control_dt_s=.01, body_names=[])
    archive = PhysicalTransitionArchive(tmp_path/'physical_transitions', metadata=metadata, compress=True)

    def state(tick, hit):
        physical = {key: torch.full((lanes, 18), float(tick)) for key in
                    ('root_position_w', 'root_quaternion_w', 'joint_position', 'joint_velocity')}
        flags = {key: torch.full((lanes,), hit, dtype=torch.bool) for key in ('had_contact', 'released')}
        return dict(**{key: torch.zeros(lanes) for key in STATE_FIELDS}, control_tick=tick,
                    episode_index=torch.zeros(lanes, dtype=torch.long),
                    physical=physical, task=flags, recovery={})

    memory = {'tensors': {path: torch.zeros(lanes, 6) for path in MIXER_STATE.values()}}
    action = torch.zeros(lanes, 18)
    for tick in range(2):
        archive.begin_record(state(tick, hit=tick > 0), action, torch.zeros(lanes, 794), memory)
        archive.pending['base_action'] = action
        end = torch.full((lanes,), tick == 1, dtype=torch.bool)
        archive.finish_record(state(tick+1, hit=True), action, action, torch.zeros(lanes), end,
                              torch.zeros_like(end), memory)
    archive.flush()
    episodes = [dict(env_id=i, episode_index=0, case_id=f'case-{i}', scoring_schema=APPROACH_BALL_SCORING,
                     numerical_failures=0, ball_side_complete_except_recovery=False,
                     physical_recovery_success=False,
                     approach_ball_terminal=dict(recovery_tail_observed=i < 2, numerical_abort=False))
                for i in range(lanes)]
    metrics = dict(task_contract=task_contract, policy_execution=execution, actor_state_sha256='actor',
                   actor_parameters_unchanged=True, optimizer_updates={'actor': 0}, control_ticks=2)
    for name, data in (
            ('episodes_summary', episodes), ('metrics', metrics), ('completed', {'episodes': lanes, 'control_ticks': 2}),
            ('startup', {'argv': ['--record-physical-transitions', '--checkpoint', 'A016.pt']})):
        (tmp_path/f'{name}.json').write_text(json.dumps(data))

    entry = index_run(tmp_path, tmp_path, frozen=True)
    assert entry['source_kind'] == 'frozen_natural'
    assert [clip['case_id'] for clip in entry['clips']] == ['case-0']
    result = export_run(entry, tmp_path)['clips'][0]
    assert not result['wm_holdout_lane'] and not result['source']['arm_intervention']
    assert result['trajectory']['terminated'].tolist() == [False, True]
    assert result['trajectory']['arm_identification_active'].tolist() == [False, False]

    (tmp_path/'startup.json').write_text(json.dumps({'argv': ['--record-physical-transitions', '--checkpoint', 'A016.pt',
                                                               '--research-fixed128']}))
    with pytest.raises(ValueError, match='unperturbed TRAIN128'):
        export_run(entry, tmp_path)
