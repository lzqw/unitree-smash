"""Enforce an audited configuration constraint at every prediction microstep.

This rejects authored hip/mount overlap; it does not synthesize a collision
impulse or change physical joint limits. Ground distances keep their units.
"""
import json
import numpy as np


class HipMountPathConstraintV1:
    def __init__(self, ground, audit, margin_rad=.02):
        if not 0<=margin_rad<=.05:raise ValueError('Declared geometric margin required')
        self.ground=ground;self.model=ground.model;self.shapes=ground.shapes
        self.calls=0;self.constraints=[];self.last_violation=None
        model=self.model
        parent=lambda name:model.model.frames[model.body_frames[name]].parent
        if parent('MountPlate')!=parent('base'):raise ValueError('Mount must be fixed to the base')
        for row in audit['angular_boundaries']:
            j=model.joint_order.index(row['joint'])
            if model.model.names[parent(row['hip'])]!=row['joint']:
                raise ValueError('Unexpected hip frame dependency')
            for boundary in row['authored_intersection_boundaries']:
                lower=boundary['overlap_on_lower_side']
                bound=boundary['angle_rad']+(margin_rad if lower else -margin_rad)
                self.constraints.append((model.q_indices[j],bound,lower,row['hip']))
        if len(self.constraints)!=2:raise ValueError('Expected the two audited front hip boundaries')

    def evaluate(self,q,*,jacobian=False):
        self.calls+=1
        for index,bound,lower,hip in self.constraints:
            violation=bound-q[index] if lower else q[index]-bound
            if violation>1e-8:
                self.last_violation=dict(hip=hip,partner='MountPlate',configuration_rad=float(q[index]),
                    planning_boundary_rad=bound,violation_rad=float(violation),
                    prediction_geometry_call=self.calls)
                raise ValueError('authored_hip_mount_constraint:'+json.dumps(self.last_violation,sort_keys=True))
        return self.ground.evaluate(q,jacobian=jacobian)
