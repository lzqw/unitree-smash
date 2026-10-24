#!/usr/bin/env python3
"""CPU high-contact pose/velocity seeds for subsequent native trajectory search."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.optimize import least_squares,lsq_linear
from quadsmash.control.full_racket_geometry_v4 import FullRacketGeometryV4


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--face-sign',type=int,choices=(-1,1),default=1)
    p.add_argument('--root-x',type=float,default=.70);p.add_argument('--root-z',type=float,default=.75)
    p.add_argument('--normal-z',type=float,default=-.10)
    p.add_argument('--ball-x',type=float,default=1.10);p.add_argument('--ball-z',type=float,default=1.80)
    p.add_argument('--point-vx',type=float,default=1.6);p.add_argument('--point-vz',type=float,default=0.)
    p.add_argument('--uv-y',type=float,default=0.);p.add_argument('--uv-z',type=float,default=0.)
    args=p.parse_args();args.output.mkdir(exist_ok=False,parents=True)
    geometry=FullRacketGeometryV4();rng=np.random.default_rng(720201)
    outward=np.array([1.,0.,args.normal_z]);outward/=np.linalg.norm(outward)
    normal=args.face_sign*outward
    ball=np.array([args.ball_x,0.,args.ball_z]);target=ball-.0256*outward
    root=np.array([args.root_x,0.,args.root_z]);bounds=geometry.bounds
    uv=(args.uv_y,args.uv_z)
    default=np.array([0.,1.,-1.,0.,0.,0.])
    reference=default if args.face_sign==1 else np.array([.0786861032,1.20102334,-2.585356,-.156469598,-1.056915283,.13533029])
    measured=json.loads((ROOT/'reports/G07_full_racket_v4/stand_regression02/trajectory.json').read_text())[0]['tool_position_m']
    initial,_=geometry.forward(default)
    error=float(np.linalg.norm(initial-measured));assert error<1e-5
    rows=[]
    def residual(q):
        contact,outward_actual=geometry.contact_center(q,uv=uv,face=args.face_sign,base_position=root)
        return np.r_[5*(contact-ball),outward_actual-outward,.001*(q-reference)]
    for index in range(32):
        initial=reference if index==0 else rng.uniform(bounds[:,0]+.001,bounds[:,1]-.001)
        fit=least_squares(residual,initial,bounds=(bounds[:,0]+1e-5,bounds[:,1]-1e-5),max_nfev=700)
        q=fit.x;point,rotation=geometry.forward(q,base_position=root)
        jacobian=geometry.contact_jacobian(q,uv=uv,face=args.face_sign,radius=.025,base_position=root)
        room=np.minimum(q-bounds[:,0],bounds[:,1]-q)
        maximum=np.maximum(np.minimum(4.8,room/.08),1e-6)
        desired_velocity=np.array([args.point_vx,0.,args.point_vz])
        speed=lsq_linear(jacobian,desired_velocity,bounds=(-maximum,maximum)).x
        contact,outward_actual=geometry.contact_center(q,uv=uv,face=args.face_sign,base_position=root)
        score=float(np.linalg.norm(contact-ball)+.2*np.linalg.norm(outward_actual-outward))
        rows.append(dict(index=index,joint_position_rad=q.tolist(),joint_velocity_seed_rad_s=speed.tolist(),
            string_center_m=point.tolist(),normal_world=rotation[:,0].tolist(),
            predicted_point_velocity_mps=(jacobian@speed).tolist(),score=score,
            planned_root_position_m=root.tolist(),desired_ball_center_m=ball.tolist(),
            face_sign=args.face_sign,outward_contact_normal_world=outward.tolist()))
        rows[-1]['desired_point_velocity_mps']=desired_velocity.tolist()
        rows[-1]['string_uv_m']=list(uv);rows[-1]['virtual_ball_center_m']=contact.tolist()
    rows.sort(key=lambda row:row['score'])
    result=dict(kind='CPU geometric seeds only; no executable trajectory or success claim',
        schema='full-racket-ik-seeds-v4',default_pose_native_fk_error_m=error,base_position_assumed=root.tolist(),
        desired_ball_center_m=ball.tolist(),desired_normal_world=normal.tolist(),seeds=rows[:12],
        all_search_results=rows,formal_acceptance=False)
    (args.output/'seeds.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'fk_error_m':error,'best':rows[0]}),flush=True)


if __name__=='__main__':main()
