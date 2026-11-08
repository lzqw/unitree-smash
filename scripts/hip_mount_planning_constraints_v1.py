"""Configuration-space constraints for the unchanged hip/mount colliders.

These angular inequalities describe self-collision geometry. They constrain
predicted joint states, not position commands or motor hardware limits. Units
remain radians, separate from the ground/net distances measured in metres.
"""
import numpy as np
from scripts.hip_mount_collision_response_v1 import HipMountCollisionResponseV1


class HipMountPlanningConstraintsV1:
    def __init__(self,model):
        self.constraints=HipMountCollisionResponseV1(model).constraints
        self.matrix=np.zeros((len(self.constraints),18));self.offset=np.zeros(len(self.constraints))
        for row,(joint,bound,lower) in enumerate(self.constraints):
            self.matrix[row,joint]=1. if lower else -1.
            self.offset[row]=-bound if lower else bound

    def gaps(self,position):
        return self.matrix@np.asarray(position)+self.offset

    def interval(self,stages):
        matrices=[];lows=[]
        for stage in stages:
            matrices.append(self.matrix@stage['joint_position_map'])
            lows.append(-self.offset-self.matrix@stage['joint_position_bias'])
        matrix=np.vstack(matrices)
        return matrix,np.concatenate(lows),np.full(len(matrix),np.inf)

    def stopping_objective(self,position,velocity,acceleration_map,acceleration_bias):
        gaps=self.gaps(position);rates=self.matrix@velocity
        active=(gaps<.15)|(rates<-.5)
        desired=np.minimum(80.,np.maximum(0.,np.maximum(
            np.maximum(0.,-rates)**2/(2*np.maximum(gaps,.005)),
            80.*(.03-gaps)-18.*rates)))
        return self.matrix[active]@acceleration_map,self.matrix[active]@acceleration_bias,desired[active]
