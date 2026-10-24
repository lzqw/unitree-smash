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
