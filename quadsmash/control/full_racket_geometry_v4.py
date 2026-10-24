"""Source-URDF full-racket kinematics; search seeds, not dynamics evidence."""
import numpy as np
from scipy.spatial.transform import Rotation
from .piper_geometry import PiperGeometry
from quadsmash.physics.full_racket_v4 import load_racket


class FullRacketGeometryV4(PiperGeometry):
    def __init__(self):
        super().__init__()
        self.racket=load_racket()
        self.tool_offset=np.array(self.racket['string_center_in_flange_m'])
        w,x,y,z=self.racket['body_rotation_in_flange_wxyz']
        self.tool_rotation=Rotation.from_quat([x,y,z,w]).as_matrix()
        self.half_extents=np.array([self.racket['assumptions']['string_thickness_m']/2,
                                  *self.racket['assumptions']['strings_semiaxes_yz_m']])

    def forward(self,q,base_position=(0.,0.,.323),base_rpy=(0.,0.,0.)):
        point,flange_rotation=super().forward(q,base_position,base_rpy)
        return point,flange_rotation@self.tool_rotation

    def contact_center(self,q,uv=(0.,0.),face=1,radius=.025,**kwargs):
        uv=np.asarray(uv,dtype=float)
        if ((uv/self.half_extents[1:])**2).sum()>1:
            raise ValueError('Point outside representative string ellipse')
        point,rotation=self.forward(q,**kwargs)
        return point+rotation@np.r_[face*(self.half_extents[0]+radius),uv],face*rotation[:,0]

    def pitch_compensation_basis(self,q,uv=(0.,0.),face=1):
        q=np.asarray(q);point,rotation=self.forward(q,base_position=(0.,0.,0.))
        offset=np.r_[face*.0256,uv];point+=rotation@offset
        columns=[]
        for axis in np.eye(6):
            hi,rhi=self.forward(q+axis*1e-6,base_position=(0.,0.,0.))
            lo,rlo=self.forward(q-axis*1e-6,base_position=(0.,0.,0.))
            columns.append(np.r_[(hi+rhi@offset-lo-rlo@offset)/2e-6,.3*(rhi[:,0]-rlo[:,0])/2e-6])
        jacobian=np.stack(columns,axis=1)
        desired=np.r_[-np.cross([0,1,0],point),-.3*np.cross([0,1,0],rotation[:,0])]
        return np.linalg.lstsq(np.vstack((jacobian,.02*np.eye(6))),np.r_[desired,np.zeros(6)],rcond=None)[0]
