"""Checkpoint-bound handoff between a resident trainer and a native evaluator."""
import hashlib
import json
import os
from time import monotonic, sleep


def publish_boundary(directory, latest, update, gpu):
    directory.mkdir()
    checkpoint = directory / 'policy.pt'
    os.link(latest, checkpoint)
    with checkpoint.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    boundary = dict(update=update, checkpoint=str(checkpoint.resolve()), checkpoint_sha256=digest,
                    training_pid=os.getpid(), gpu=gpu)
    temporary = directory / 'ready.tmp'
    temporary.write_text(json.dumps(boundary, indent=2) + '\n')
    temporary.replace(directory / 'ready.json')
    return boundary


def await_native_decision(directory, boundary, timeout_s, check_stop):
    """Retain the scene until a matching result arrives or stop safely."""
    deadline = monotonic() + timeout_s
    path = directory / 'decision.json'
    while monotonic() < deadline:
        check_stop()
        if path.is_file():
            decision = json.loads(path.read_text())
            with (directory / 'paired.json').open('rb') as stream:
                paired_digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if (decision.get('checkpoint_sha256') != boundary['checkpoint_sha256']
                    or decision.get('update') != boundary['update']
                    or decision.get('action') not in ('continue', 'stop')
                    or decision.get('paired_result_sha256') != paired_digest):
                raise ValueError('Resident decision does not match the published Approach boundary')
            return decision['action']
        sleep(min(1., max(0., deadline - monotonic())))
    return 'stop'
