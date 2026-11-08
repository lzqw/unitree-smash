"""Native, explicitly unadmitted timed-reference recovery proposal.

Uses the existing verified dispatch/history path, with a separate constructor
for the new reference contract. It does not reinterpret a failed model plan
as a feasible teacher or claim online MPC replanning.
"""
from functools import partial
import hashlib
import json
from pathlib import Path

import numpy as np
import pinocchio as pin

from scripts.native_experimental_recovery_v2 import NativeExperimentalRecoveryV2
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.loaded_pinocchio_model_v1 import PublicState
from scripts.search_preserved_early_support_v1 import joint_factory
from scripts.timed_whole_body_recovery_v1 import build_timed_plan


class NativeTimedExperimentalRecoveryV1(NativeExperimentalRecoveryV2):
    def __init__(self,bundle,case,limits,policy_sha256,output):
        self.bundle=Path(bundle);self.output=Path(output)
        self.manifest=json.loads((self.bundle/'manifest.json').read_text());m=self.manifest
        if (m['contract']!='timed_model_rejected_native_diagnostic_v1'
                or m['model_accepted'] or m['valid_expert_labels']
                or m['policy_sha256']!=policy_sha256 or case!=m['case']):
            raise ValueError('Explicit rejected timed proposal required')
        for path,digest in m['source_sha256'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('Changed proposal source '+path)
        archive=self.bundle/m['bundle_name']
        if hashlib.sha256(archive.read_bytes()).hexdigest()!=m['bundle_sha256']:raise ValueError('Changed reference')
        self.case=int(case);self.start=280;self.steps=520;self.applied=0
        self.previous=None;self.last_tick=-1;self.records=[];self.query_check=None;self.early_termination_tick=None
        self.limits=np.asarray(limits.detach().cpu(),float)
        if not np.allclose(self.limits,m['configuration']['interface']['limits'],atol=1e-7,rtol=0):
            raise ValueError('Changed native limits')
        self.controller=build_controller(m['configuration']);self.model=self.controller.qp.model
        if m['configuration']['arm_target_rate_rad_s']!=10.:raise ValueError('Changed arm interface')
        q=np.asarray(m['query']['q'],float).copy();v=np.asarray(m['query']['v'],float).copy()
        self.initial=PublicState(q,v,pin.Quaternion(q[3:7]).matrix(),q[:3].copy(),
            q[self.model.q_indices].copy(),v[self.model.v_indices].copy(),np.asarray(m['query']['contacts'],bool),0.)
        self.initial.previous_command_target=np.asarray(m['query']['target'],float).copy()
        with np.load(archive,allow_pickle=False) as z:
            rq=z['tracking_reference_q'].copy();rv=z['tracking_reference_v'].copy()
            self.query_public=z['query_public'].copy();self.query_previous_action=z['previous_action'].copy()
        worker=dict(controller=self.controller,initial=self.initial,
            factory=partial(joint_factory,reference_q=rq,reference_v=rv,gains=m['tracking_gains']))
        self.feedback=build_timed_plan(worker,m['proposal'])
