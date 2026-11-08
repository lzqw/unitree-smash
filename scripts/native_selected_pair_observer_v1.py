"""Retarget the existing passive pair observer to a declared diagnostic window.

The observer is configured before any physics step. Ground and foot-pair
filters remain separate; no force is applied and no controller input changes.
Parent-trajectory equivalence must still be checked after native execution.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.native_ground_pair_observer_v1 import NativeGroundPairObserverV1


class NativeSelectedPairObserverV1(NativeGroundPairObserverV1):
    def __init__(self, env, output, *, lanes=(71,125), first_frame=50880, last_frame=52480):
        if (len(lanes)!=2 or len(set(lanes))!=2 or any(type(i) is not int or not 0<=i<128 for i in lanes)
                or type(first_frame) is not int or type(last_frame) is not int
                or not 0<first_frame<last_frame<=128000 or last_frame-first_frame>2400):
            raise ValueError('Two original lanes and a bounded physics-rate window required')
        super().__init__(env,output)
        if self.count or self.ground_count or self.buffers or self.filtered_buffers:
            raise ValueError('Cannot retarget an observer after capture begins')
        self.lanes=list(lanes);self.indices=torch.tensor(self.lanes,device=env.device)
        self.origins=env.scene.env_origins[self.indices].detach().cpu().numpy().copy()
        self.first,self.last=first_frame,last_frame;self.capacity=last_frame-first_frame+1
        self.frames=np.empty(self.capacity,dtype=np.int64)
        self.pair_views=[];self.pair_filters=[]
        sim=env.feet_contact._physics_sim_view
        for case in self.lanes:
            prefix=f'/World/envs/env_{case}/Robot/Go2/'
            for name in self.names:
                filters=[self.ground_path]+[prefix+other for other in self.names if other!=name]
                view=sim.create_rigid_contact_view(prefix+name,filter_patterns=filters,max_contact_data_count=64)
                if list(view.sensor_paths)!=[prefix+name] or view.filter_count!=4:
                    raise ValueError('Unexpected selected-pair view')
                self.pair_views.append(view);self.pair_filters.append(filters)

    def save(self):
        metadata=super().save()
        metadata.update(selected_observer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            declared_first_physics_frame=self.first,declared_last_physics_frame=self.last,
            retargeted_before_physics=True)
        path=self.output/'filtered_contact_observation.json';temp=path.with_suffix('.tmp')
        temp.write_text(json.dumps(metadata,indent=2)+'\n');temp.replace(path)
        return metadata
