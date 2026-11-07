"""Resolution sensitivity of a fixed controller, without retuning its reference.

Equal integration step count does not make the contact solver equal to PhysX.
All failures stay failures under the original path thresholds.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import search_terminal_prefix_arm_margin_v1 as search
from scripts import stress_terminal_support_commands_v1 as stress

ORIGINAL_PREDICT = None


def initialize(*paths):
    global ORIGINAL_PREDICT
    search.initialize(*paths)
    ORIGINAL_PREDICT = stress.predict_frictional_spheres


def evaluate(task):
    def predict(*args, **kwargs):
        return ORIGINAL_PREDICT(*args, substeps=task['substeps'], **kwargs)
    stress.predict_frictional_spheres = predict
    return search.evaluate(task)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['bundle', 'warm-plan', 'validation', 'sphere-audit', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve prior evidence')
    paths = tuple(str(path) for path in [args.bundle,args.warm_plan,args.validation,args.sphere_audit])
    initialize(*paths)
    files = [Path(__file__),args.bundle/'manifest.json',args.warm_plan,args.validation,args.sphere_audit]
    files += [Path(value.__file__) for key,value in list(sys.modules.items())
              if key.startswith('scripts.') and getattr(value,'__file__',None)]
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    hashes = {str(path.resolve()):digest(path) for path in files}
    tasks = [dict(name='nominal',feedback=True,start=0,end=0,joint_start=0,joint_end=4,
                  amplitude_rad=0.,arm_correction_rad=correction,substeps=substeps)
             for correction in [0.,-.006] for substeps in [16,32,64,160]]
    rows=[]
    started=time.monotonic()
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize,initargs=paths) as pool:
        for row in pool.map(evaluate,tasks):
            rows.append(row)
            print(json.dumps(row),flush=True)
            temporary=args.output.with_suffix('.partial.tmp')
            temporary.write_text(json.dumps(dict(rows=rows,complete=False),indent=2)+'\n')
            temporary.replace(args.output.with_suffix('.partial.json'))
    for path,expected in hashes.items():
        if digest(path)!=expected:
            raise ValueError('Input changed: '+path)
    output=dict(scope=__doc__,rows=rows,complete=True,source_sha256=hashes,
                wall_seconds=time.monotonic()-started,native_execution=False,
                valid_expert_labels=False,physx_equivalence_proven=False)
    temporary=args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    temporary.replace(args.output)


if __name__=='__main__':
    main()
