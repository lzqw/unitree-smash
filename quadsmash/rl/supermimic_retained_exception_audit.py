"""Read-only shutdown diagnostics: names/types, never tensor values or mutation."""
import gc
import json
from pathlib import Path


def _type_name(value):
    cls = type(value)
    return cls.__module__ + '.' + cls.__qualname__


def _frame_summary(frame):
    return dict(file=frame.f_code.co_filename, function=frame.f_code.co_name,
                line=frame.f_lineno,
                local_types={str(k): _type_name(v) for k, v in frame.f_locals.items()})


def _owners(value, ignored):
    result = []
    for owner in gc.get_referrers(value):
        if id(owner) in ignored:
            continue
        row = {'type': _type_name(owner)}
        if type(owner) is dict:
            row['matching_keys'] = [key for key, item in owner.items()
                                    if type(key) is str and item is value]
            if type(owner.get('__name__')) is str:
                row['module'] = owner['__name__']
        elif type(owner) in (tuple, list):
            row['length'] = len(owner)
        result.append(row)
    return result


def capture_retained_exceptions(output):
    """Inspect caught exceptions before Kit teardown; do not clear tracebacks.

    Inspection itself temporarily retains references/materializes frame locals;
    all are discarded on return. This does not prove that all native resources
    are tracked by Python GC. Exception messages/locals' values are not dumped.
    """
    objects = gc.get_objects()
    errors = [item for item in objects
              if isinstance(item, BaseException) and item.__traceback__ is not None]
    del objects
    records = []
    for error in errors:
        trace, frames = error.__traceback__, []
        while trace is not None and len(frames) < 64:
            frame = trace.tb_frame
            backs, back = [], frame.f_back
            while back is not None and len(backs) < 32:
                backs.append(_frame_summary(back))
                back = back.f_back
            frames.append(dict(traceback_line=trace.tb_lineno, frame=_frame_summary(frame),
                               parents=backs))
            trace = trace.tb_next
        records.append(dict(type=_type_name(error),
                            owners=_owners(error, {id(errors)}), frames=frames))
    payload = dict(scope='Read-only retained Python exception/frame metadata before app.close',
                   exceptions=records, count=len(records))
    (Path(output)/'retained_exceptions.json').write_text(json.dumps(payload, indent=2)+'\n')
    return len(records)
