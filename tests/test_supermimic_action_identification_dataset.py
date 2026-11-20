import json

import pytest

from quadsmash.rl.supermimic_action_identification_dataset_v1 import pack_collection


def test_incomplete_collection_is_rejected(tmp_path):
    (tmp_path / "status.json").write_text(json.dumps({"status": "RUNNING"}))
    with pytest.raises(ValueError, match="not complete"):
        pack_collection(tmp_path, tmp_path / "out.pt")


def test_empty_collection_is_rejected(tmp_path):
    (tmp_path / "status.json").write_text(json.dumps({"status": "COLLECTION_FINISHED"}))
    with pytest.raises(ValueError, match="no completed"):
        pack_collection(tmp_path, tmp_path / "out.pt")
