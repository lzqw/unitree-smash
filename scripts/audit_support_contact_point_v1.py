"""Compare foot-frame and authored sphere contact-point support kinematics.

Uses actual native states and ground-only forces. Velocity differences show
the modeling distinction, not proof that it causes a native landing failure.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
from scripts.predict_self_contact_spheres_v2 import surface_jacobians


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();read=lambda p:json.loads(Path(p).read_text())
    if a.output.exists():raise ValueError('Fresh output required')
    with np.load(a.run/'pure_control_commands.npz') as z:d={k:z[k].copy() for k in z.files}
    with np.load(a.run/'pure_control_ground_forces.npz') as z:peaks=z['interval_peak_normal_n'].copy()
    meta=read(a.run/'pure_control_ground_forces.json');model=LoadedPinocchioModelV1()
    spheres=read(ROOT/'configs/go2_piper_authored_foot_spheres_v1.json')['spheres']
    nonfeet=[i for i,n in enumerate(meta['body_names']) if n not in model.feet]
    result=[]
    for i in range(len(d['q'])):
        support=d['ground_force'][i,:,2]>=5.
        if not support.any():continue
        tick=int(d['tick'][i]);lane=int(d['lane'][i])
        if tick==0 or peaks[tick-1,lane,nonfeet].max()>1.:continue
        q=d['q'][i];v=d['v'][i];snapshot=model.snapshot(q,v)
        center=snapshot['foot_jacobian']@v;contact=surface_jacobians(model,q,spheres)@v
        for foot in np.flatnonzero(support):
            result.append(dict(case=lane,tick=tick,foot=model.feet[foot],
                normal_force_n=float(d['ground_force'][i,foot,2]),
                foot_frame_velocity_m_s=center[foot].tolist(),
                sphere_surface_velocity_m_s=contact[foot].tolist(),
                velocity_difference_m_s=float(np.linalg.norm(contact[foot]-center[foot])),
                frame_tangent_speed_m_s=float(np.linalg.norm(center[foot,:2])),
                surface_tangent_speed_m_s=float(np.linalg.norm(contact[foot,:2]))))
    def stats(name):
        v=[r[name] for r in result]
        return dict(median=float(np.median(v)),p95=float(np.percentile(v,95)),maximum=float(max(v))) if v else None
    summary=dict(samples=len(result),cases=len(set(r['case'] for r in result)),
        velocity_difference=stats('velocity_difference_m_s'),frame_tangent_speed=stats('frame_tangent_speed_m_s'),
        surface_tangent_speed=stats('surface_tangent_speed_m_s'),
        force_threshold_diagnostic_n=5.,prior_interval_nonfoot_max_n=1.,
        native_control_changed=False,causal_recovery_gain_not_established=True)
    paths=[Path(__file__),a.run/'pure_control_commands.npz',a.run/'pure_control_ground_forces.npz',ROOT/'configs/go2_piper_authored_foot_spheres_v1.json']
    output=dict(summary=summary,samples=result,source_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
