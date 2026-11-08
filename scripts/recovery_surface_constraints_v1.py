"""Shared authored ground, net and foot-pair distances for recovery planning.

The interval QP receives the same distances as its integrated path check.
Net rows are exact nearby convex witnesses; far rows are positive conservative
lower bounds with no asserted derivative. This is not a PhysX collision model
or a complete self-collision certificate. Native geometry remains unchanged.
"""
import itertools
import numpy as np
from scripts.authored_net_clearance_v1 import AuthoredNetClearanceV1
from scripts.foot_pair_clearance_kinematics_v1 import foot_pair_clearance
from quadsmash.physics.court_v4 import load_court


class RecoverySurfaceConstraintsV1:
    def __init__(self,ground,spheres):
        self.ground=ground;self.model=ground.model;self.spheres=spheres
        self.net=AuthoredNetClearanceV1(ground,load_court())
        self.net_names=list(self.net.bodies)
        self.pairs=list(itertools.combinations(self.model.feet,2))
        self.shapes=list(ground.shapes)+[dict(body=n+'::net',constraint='authored_net') for n in self.net_names]
        self.shapes += [dict(body=a+'::'+b,constraint='foot_pair') for a,b in self.pairs]
        self.last_evidence={}

    def evaluate(self,q,*,jacobian=False):
        gaps,jacs=self.ground.evaluate(q,jacobian=jacobian)
        # A fresh Jacobian calculation is required when derivatives are used;
        # the non-derivative path reuses the ground model's current placements.
        net=self.net.evaluate(q,jacobian=jacobian,placements_current=not jacobian,near_distance_m=.20)
        ng=[];nj=[]
        for name in self.net_names:
            row=net[name];ng.append(row['distance_m'])
            nj.append(row.get('jacobian',np.zeros(self.model.model.nv)) if row['exact'] else np.zeros(self.model.model.nv))
        pg,pj=foot_pair_clearance(self.model,q,self.spheres)
        all_gaps=np.concatenate([gaps,ng,pg])
        all_jacs=np.vstack([jacs,nj,pj]) if jacobian else None
        self.last_evidence=dict(minimum_ground_m=float(np.min(gaps)),minimum_net_m=float(min(ng)),
            minimum_foot_pair_m=float(pg.min()),net_exact_rows=sum(r['exact'] for r in net.values()))
        if not np.isfinite(all_gaps).all() or (jacobian and not np.isfinite(all_jacs).all()):
            raise ValueError('Nonfinite recovery surface constraint')
        return all_gaps,all_jacs
