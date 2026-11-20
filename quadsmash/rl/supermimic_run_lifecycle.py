"""Durable native shutdown evidence; no Isaac dependency or forced process exit."""
import json
import os
from pathlib import Path
import time
import traceback
import gc
import sys


def stage(output, name, **fields):
    with (Path(output) / 'lifecycle.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(stage=name, pid=os.getpid(), time_ns=time.time_ns(), **fields))+'\n')
        stream.flush()
        os.fsync(stream.fileno())


def close_resource(output, name, resource, *, primary_failed):
    if resource is None:
        return
    stage(output, 'BEFORE_' + name + '_CLOSE')
    try:
        resource.close()
    except BaseException as error:
        (Path(output)/(name.lower() + '_cleanup_failure.txt')).write_text(traceback.format_exc())
        stage(output, name + '_CLOSE_FAILED', error_type=type(error).__name__, message=str(error))
        if not primary_failed:
            raise
    else:
        stage(output, 'AFTER_' + name + '_CLOSE')


def execute_scoped_native(output, run, app):
    """Release simulation locals/cycles while Kit is still alive, then close it.

    `run` must close its environment and must not return simulation objects.
    Exception tracebacks are saved by the caller before clearing frame locals.
    """
    failed = False
    try:
        run()
    except BaseException as error:
        failed = True
        # An unwound traceback otherwise keeps native objects alive past Kit.
        (Path(output)/'scoped_failure_traceback.txt').write_text(traceback.format_exc())
        traceback.clear_frames(error.__traceback__)
        raise
    finally:
        release_kit_stat_cache(output)
        stage(output, 'BEFORE_NATIVE_GC')
        collected = gc.collect()
        stage(output, 'AFTER_NATIVE_GC', collected=collected)
        if os.environ.get('QUADSMASH_AUDIT_RETAINED_EXCEPTIONS') == '1':
            from .supermimic_retained_exception_audit import capture_retained_exceptions
            count = capture_retained_exceptions(output)
            stage(output, 'RETAINED_EXCEPTION_AUDIT_FINISHED', count=count)
        close_native(output, None, app, primary_failed=failed)


def native_process_main(main):
    """Exit a process-owned Kit 5.1 run after scoped cleanup and preserve failures."""
    os.environ['QUADSMASH_SKIP_KIT_CLOSE'] = '1'
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def release_kit_stat_cache(output):
    """Reset the identified import cache while PhysX/Kit are still alive.

    Kit caches OSError instances and re-raises them. Their traceback frames can
    keep an already closed environment's tensor views alive until interpreter
    shutdown. Use Kit's existing reset API; do not mutate exception frames or
    patch installed code. Do not import Kit in CPU/non-Kit processes.
    """
    module = sys.modules.get('omni.ext._impl.stat_cache')
    if module is None:
        stage(output, 'KIT_STAT_CACHE_NOT_LOADED')
        return
    cache = getattr(module, '_stat_cache', None)
    reset = getattr(module, 'reset_stat_cache', None)
    if not isinstance(cache, dict) or not callable(reset):
        raise RuntimeError('Installed Kit stat-cache API changed; refusing silent cleanup bypass')
    before = len(cache)
    errors_before = sum(isinstance(value, tuple) and len(value) == 2
                        and isinstance(value[1], BaseException) for value in cache.values())
    stage(output, 'BEFORE_KIT_STAT_CACHE_RESET', entries=before, exceptions=errors_before)
    reset()
    after = len(module._stat_cache)
    stage(output, 'AFTER_KIT_STAT_CACHE_RESET', entries=after)
    if after:
        raise RuntimeError('Kit stat-cache reset retained entries; see lifecycle evidence')


def close_native(output, env, app, *, primary_failed):
    errors = []
    # Isaac Sim 5.1 can abort inside SimulationApp.close() while PhysX is
    # releasing its framework (and can also abort later during Python's module
    # finalizers).  The caller uses os._exit after this scope, so on that
    # process-owned path the OS can reclaim Kit safely without invoking the
    # broken shutdown sequence.
    resources = [('ENV', env)]
    if os.environ.get('QUADSMASH_SKIP_KIT_CLOSE') != '1':
        resources.append(('APP', app))
    else:
        stage(output, 'KIT_CLOSE_SKIPPED_FOR_PROCESS_EXIT')
    for name, resource in resources:
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


def inspect_process_result(output, returncode, *, shutdown_diagnostic_ticks=None):
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
    diagnostic_completed = False
    if shutdown_diagnostic_ticks is not None:
        try:
            payload = json.loads((output/'shutdown_diagnostic.json').read_text())
            diagnostic_completed = (
                1 <= shutdown_diagnostic_ticks < 800
                and payload.get('control_ticks') == shutdown_diagnostic_ticks
                and payload.get('requested_ticks') == shutdown_diagnostic_ticks
                and payload.get('num_envs') == 128
                and not completed.exists() and not (output/'metrics.json').exists())
        except (OSError, ValueError, TypeError):
            pass
    work_completed = simulation_completed if shutdown_diagnostic_ticks is None else diagnostic_completed
    valid = returncode == 0 and work_completed and cleanup_finished and not (output/'failure.json').exists()
    return dict(child_returncode=returncode, simulation_completed=simulation_completed,
                shutdown_diagnostic_completed=diagnostic_completed,
                run_kind='formal_128' if shutdown_diagnostic_ticks is None else 'shutdown_diagnostic',
                cleanup_finished=cleanup_finished, process_completed_cleanly=valid,
                diagnostic_exit_code=0 if valid else (128-returncode if returncode < 0 else returncode or 1))
