"""Full-authority event feedback with net/foot-pair-aware interval execution.

This is a native experimental WBC candidate, NOT sequence MPC or a validated
landing teacher. It removes legacy arm command slew, adds physical clearance
rows to planning and checks the same geometry after interval integration.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from scripts.pure_control_recovery_v1 import PureControlRecoveryV1,ROOT
from scripts.full_authority_position_interface_v1 import FullAuthorityPositionPDInterfaceV1
from scripts.recovery_surface_constraints_v1 import RecoverySurfaceConstraintsV1


@lru_cache(maxsize=1)
def authored_feet(asset,digest,feet):
    path=Path(asset)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Changed original robot asset')
    record=json.loads((ROOT/'configs/go2_piper_authored_foot_spheres_v1.json').read_text())
    for dependency,expected in record['asset_sha256'].items():
        if hashlib.sha256(Path(dependency).read_bytes()).hexdigest()!=expected:raise ValueError('Changed foot asset dependency')
    if [s['body'] for s in record['spheres']]!=list(feet):raise ValueError('Wrong authored foot order')
    return record['spheres']


def _pinned_asset_digest(source_sha256, asset):
    """Resolve a pinned asset by content when an old worktree path is recorded.

    The asset bytes and hash remain authoritative; only the absolute path key is
    made portable across worktrees. Ambiguous or mismatched content still fails.
    """
    asset = Path(asset)
    actual = hashlib.sha256(asset.read_bytes()).hexdigest()
    exact = source_sha256.get(str(asset))
    if exact is not None:
        if exact != actual:
            raise ValueError('Changed original robot asset')
        return exact
    matches = [expected for recorded, expected in source_sha256.items()
               if Path(recorded).name == asset.name and expected == actual]
    if len(matches) != 1:
        raise ValueError('No unique pinned digest for current-worktree asset: '+str(asset))
    return matches[0]


class PureControlCourtRecoveryV4(PureControlRecoveryV1):
    def __init__(self,interface,nominal_posture):
        super().__init__(interface,nominal_posture)
        self.interface=FullAuthorityPositionPDInterfaceV1(interface)
        asset=ROOT/'assets/quadsmash_full_racket_v4_r3.usd'
        digest=_pinned_asset_digest(self.model.record['source_sha256'], asset)
        spheres=authored_feet(str(asset),digest,tuple(self.model.feet))
        self.geometry=RecoverySurfaceConstraintsV1(self.geometry,spheres)

    def command(self,state,phase,time_s):
        target,diagnostic=super().command(state,phase,time_s)
        diagnostic.update(controller='pure_control_court_recovery_v4',
            software_arm_command_slew_removed=True,physical_actuator_parameters_unchanged=True,
            physical_surface_evidence=self.geometry.last_evidence.copy(),
            interval_surface_minimum_includes_ground_net_and_foot_pairs=True,
            fallback_still_experimental=not diagnostic['solver_accepted'])
        return target,diagnostic
