import numpy as np
import pytest
from quadsmash.rl.paired_native_recovery_dataset_v1 import paired_labels


def sample():
    public=np.zeros((4,794),np.float32);public[:,775]=np.arange(4)*.01
    public[1,773]=1;public[2:,774]=1
    base=np.zeros((4,18),np.float32);student=base.copy();final=base.copy()
    base[2:,0]=.1;student[2:,0]=.4;final[2:,0]=.7
    previous=np.vstack((np.zeros((1,18),np.float32),final[:-1]))
    active=np.array([False,False,True,True]);limits=np.tile([-2.,2.],(18,1))
    return public,base,student,final,previous,active,limits


def test_total_label_preserves_student_plus_teacher_correction():
    labels=paired_labels(*sample(),recovery_success=True,complete5=True)
    assert labels['supervision_mask'].tolist()==[False,False,True,True]
    np.testing.assert_allclose(labels['replacement_delta'][2:,0],.3)
    np.testing.assert_allclose(labels['measured_teacher_increment'][2:,0],.15)
    np.testing.assert_allclose(labels['replacement_delta'][2:],
        labels['measured_teacher_increment'][2:]+labels['measured_student_delta'][2:])
    assert labels['reconstruction_error_max_rad']<1e-7


def test_failed_correction_never_becomes_positive():
    labels=paired_labels(*sample(),recovery_success=False,complete5=False)
    assert not labels['supervision_mask'].any()
    assert not labels['replacement_delta'].any()
    assert labels['teacher_changed'].any()


def test_recovery_only_is_usable_without_becoming_complete_success():
    labels=paired_labels(*sample(),recovery_success=True,complete5=False)
    assert labels['supervision_mask'].sum()==2
    with pytest.raises(ValueError):paired_labels(*sample(),recovery_success=False,complete5=True)


def test_unexecuted_history_and_impossible_arm_labels_are_rejected():
    data=list(sample());data[4]=np.zeros_like(data[4])
    with pytest.raises(ValueError):paired_labels(*data,recovery_success=True,complete5=True)
    data=list(sample());data[3][2,12]=.4;data[4][3]=data[3][2]
    with pytest.raises(ValueError):paired_labels(*data,recovery_success=True,complete5=True)


def test_loader_keeps_original_base_in_features_and_total_target(tmp_path):
    import hashlib
    import json
    from quadsmash.rl.paired_native_recovery_dataset_v1 import SCHEMA,FEATURES,load_paired_episodes
    public,base,student,final,previous,active,limits=sample()
    labels=paired_labels(public,base,student,final,previous,active,limits,recovery_success=True,complete5=True)
    path=tmp_path/'case_000.npz'
    np.savez_compressed(path,packed_public=np.concatenate((public[:,:69],public[:,759:]),axis=1),
        original_base_command=base,student_command=student,final_command=final,previous_final_command=previous,
        recovery_eligible=active,supervision_mask=labels['supervision_mask'],replacement_delta=labels['replacement_delta'])
    manifest=dict(schema=SCHEMA,feature_semantics=FEATURES,joint_limits_rad=limits.tolist(),positive_episodes=1,
        episodes=[dict(case=0,complete5=True,outcome='complete5',data_file=path.name,
            data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            full_public794_sha256=hashlib.sha256(public.tobytes()).hexdigest())])
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    _,episodes,_=load_paired_episodes(tmp_path);episode=episodes[0]
    np.testing.assert_allclose(episode['features'][:,794],.1)
    np.testing.assert_allclose(episode['delta'][:,0],.3)
    np.testing.assert_allclose(episode['target'][:,0],.7)
