"""Set this process's loaded OpenBLAS pools after Isaac startup.

Isaac SimulationApp overwrites OPENBLAS_NUM_THREADS. Small interval KKT solves
can otherwise create a large SciPy pool after startup. No physics/Torch pool,
other process, package file or machine-wide configuration is modified here.
"""
import ctypes
import os
from pathlib import Path


def configure_landing_blas_threads(count=1):
    if type(count) is not int or not 1 <= count <= 32:
        raise ValueError('Explicit BLAS thread count in 1..32 required')
    # Future BLAS imports should inherit the same setting.
    os.environ['OPENBLAS_NUM_THREADS'] = str(count)
    maps = Path('/proc/self/maps').read_text().splitlines()
    libraries = sorted({line.split()[-1] for line in maps
                        if 'libopenblas' in line or 'libscipy_openblas' in line})
    records = []
    for path in libraries:
        lib = ctypes.CDLL(path)
        names = [('scipy_openblas_get_num_threads', 'scipy_openblas_set_num_threads'),
                 ('openblas_get_num_threads64_', 'openblas_set_num_threads64_'),
                 ('openblas_get_num_threads', 'openblas_set_num_threads')]
        pair = next(((getattr(lib,g),getattr(lib,s)) for g,s in names
                     if hasattr(lib,g) and hasattr(lib,s)), None)
        if pair is None:
            raise ValueError('Unsupported loaded OpenBLAS API: ' + path)
        getter,setter = pair
        getter.argtypes=[];getter.restype=ctypes.c_int
        setter.argtypes=[ctypes.c_int];setter.restype=None
        before=int(getter());setter(count);after=int(getter())
        if after != count:
            raise ValueError('OpenBLAS thread limit did not apply: ' + path)
        records.append(dict(library=path,before=before,after=after))
    if not records:
        raise ValueError('No loaded OpenBLAS pool; call after numerical imports')
    return dict(schema='landing-openblas-runtime-v1',requested_threads=count,pools=records,
                physics_threads_unchanged=True,physical_parameters_unchanged=True)
