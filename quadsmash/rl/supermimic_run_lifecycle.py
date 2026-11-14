"""Durable native shutdown evidence; no Isaac dependency or forced process exit."""
import json
import os
from pathlib import Path
import time
import traceback


def stage(output, name, **fields):
    with (Path(output) / 'lifecycle.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(stage=name, pid=os.getpid(), time_ns=time.time_ns(), **fields))+'\n')
        stream.flush()
        os.fsync(stream.fileno())


def close_native(output, env, app, *, primary_failed):
    errors = []
    for name, resource in (('ENV', env), ('APP', app)):
        if resource is None:
            continue
        stage(output, 'BEFORE_' + name + '_CLOSE')
        try:
            resource.close()
        except BaseException as error:
            trace = traceback.format_exc()
            (Path(output) / (name.lower() + '_cleanup_failure.txt')).write_text(trace)
            stage(output, name + '_CLOSE_FAILED', error_type=type(error).__name__, message=str(error))
            errors.append(error)
        else:
            stage(output, 'AFTER_' + name + '_CLOSE')
    if errors:
        # Preserve the original simulation exception if one is propagating.
        if not primary_failed:
            raise RuntimeError('Native cleanup failed; see lifecycle and cleanup traceback') from errors[0]
    else:
        stage(output, 'CLEANUP_FINISHED')


def inspect_process_result(output, returncode):
    """Process exit and completed simulation are independently necessary."""
    output = Path(output)
    completed = output / 'completed.json'
    try:
        payload = json.loads(completed.read_text())
        simulation_completed = payload.get('control_ticks') == 800 and payload.get('episodes') == 128
    except (OSError, ValueError, TypeError):
        simulation_completed = False
    try:
        events = [json.loads(line) for line in (output/'lifecycle.jsonl').read_text().splitlines()]
        cleanup_finished = bool(events and events[-1]['stage'] == 'CLEANUP_FINISHED')
    except (OSError, ValueError, KeyError, TypeError):
        cleanup_finished = False
    valid = returncode == 0 and simulation_completed and cleanup_finished and not (output/'failure.json').exists()
    return dict(child_returncode=returncode, simulation_completed=simulation_completed,
                cleanup_finished=cleanup_finished, process_completed_cleanly=valid,
                diagnostic_exit_code=0 if valid else (128-returncode if returncode < 0 else returncode or 1))
