"""Native shuttle flight and force-confirmed robot/racket collision events."""
from __future__ import annotations
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass
from isaacsim.core.utils.stage import get_current_stage

from quadsmash.physics.native import (physics_contract, drag_force, configure_ccd,
    sphere_box_gap, swept_sphere_box, segment_plane_crossing, ContactEpisodes)
from .jump_env import JumpEnv, JumpEnvCfg
from .coupled_env import CoupledEnv

PHYSICS = physics_contract()
NET_GEOMETRY_VERSION = 'environment-local-net-v2'


@configclass
class BallEnvCfg(JumpEnvCfg):
    net_geometry_version = NET_GEOMETRY_VERSION
    event_goal_enabled = True
    # G06 development contact panel: this is the validated airborne G05
    # racket event, exposed through the same native command fields.  The
    # shuttle is launched once at reset and then integrated by PhysX; it is
    # never corrected near the racket.
    event_goal_time_s = 1.48
    event_goal_position = (0.243, 0.001, 0.553)
    event_goal_normal = (-0.525, 0.0, -0.851)
    event_goal_velocity = (0.0, 0.0, 0.0)
    episode_length_s = 4.0
    sim = sim_utils.SimulationCfg(dt=PHYSICS["native_physics_dt_s"], render_interval=160)
    decimation = round(PHYSICS["policy_dt_s"]/PHYSICS["native_physics_dt_s"])
    shuttle_radius_m = PHYSICS["collision_radius_m"]
    shuttle_mass_kg = PHYSICS["mass_kg"]
    drag_k_per_m = PHYSICS["quadratic_drag_k_per_m"]
    wind_m_s = tuple(PHYSICS["wind_m_s"])
    shuttle_start = (7.4, 0.0, 1.15)
    # A physical high launch produces a natural crossing of the fixed G05
    # contact panel near 1.48 s under the locked drag/gravity model.
    shuttle_velocity = (-6.0, 0.0, 6.235)
    include_net = False
    net_height_m = 1.524
    net_side_height_m = 1.524
    net_profile_segments = 32
    net_x_m = 1.5
    target_x_min_m = 2.0
    target_x_max_m = 4.0
    target_y_half_width_m = 1.0
    shuttle = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Shuttle",
        spawn=sim_utils.SphereCfg(
            radius=PHYSICS["collision_radius_m"],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, linear_damping=0.0, angular_damping=0.0,
                max_linear_velocity=PHYSICS["max_speed_m_s"],
                solver_position_iteration_count=8, solver_velocity_iteration_count=2),
            mass_props=sim_utils.MassPropertiesCfg(mass=PHYSICS["mass_kg"]),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=PHYSICS["contact_offset_m"], rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                restitution=PHYSICS["restitution"], static_friction=PHYSICS["friction"],
                dynamic_friction=PHYSICS["friction"], restitution_combine_mode="max"),
            activate_contact_sensors=True),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(7.4, 0, 1.15), lin_vel=(-6.0, 0, 6.235)))
    shuttle_contact = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Shuttle", update_period=0.0,
        track_contact_points=True, max_contact_data_count_per_prim=16,
        filter_prim_paths_expr=["/World/envs/env_.*/Robot/Racket", "/World/Ground/geometry/mesh"])


class BallEnv(JumpEnv):
    cfg: BallEnvCfg

    def __init__(self, cfg, **kwargs):
        self.contact_events = None
        cfg.sim.physx.enable_ccd = False
        cfg.sim.physx.bounce_threshold_velocity = 0.05
        super().__init__(cfg, **kwargs)
        self.racket_body_id = self.robot.find_bodies("Racket")[0][0]
        self.foot_ids = self.robot.find_bodies(".*_foot")[0]
        self.shuttle_contact = self.scene.sensors["shuttle_contact"]
        n, device = self.num_envs, self.device
        self.contact_events = ContactEpisodes(n, device, PHYSICS["contact_force_threshold_n"], PHYSICS["separation_hysteresis_m"])
        for name in ("contact_seen", "airborne_contact_seen", "body_contact_seen", "crossed_net", "landed_target", "landed"):
            setattr(self, name, torch.zeros(n, dtype=torch.bool, device=device))
        for name in ("contact_impulse", "ball_clock", "outgoing_speed", "outgoing_vz", "_contact_bonus"):
            setattr(self, name, torch.zeros(n, device=device))
        self.contact_time = torch.full((n,), float("nan"), device=device)
        self.ball_min_distance = torch.full((n,), float("inf"), device=device)
        self.ball_min_distance_time = torch.zeros(n, device=device)
        self.ball_min_distance_ball = torch.zeros((n, 3), device=device)
        self.ball_min_distance_tool = torch.zeros((n, 3), device=device)
        self.previous_ball_velocity = torch.zeros((n, 3), device=device)
        self.previous_ball_position = torch.zeros((n, 3), device=device)
        self.previous_drag = torch.zeros((n, 3), device=device)
        self._latest_racket_force = torch.zeros((n, 3), device=device)
        self.previous_tool_position = torch.zeros((n,3), device=device)
        self.previous_tool_quaternion = torch.zeros((n,4), device=device)
        self.previous_tool_quaternion[:,0] = 1.
        self.sweep_ready = torch.zeros(n, dtype=torch.bool, device=device)
        self.sweep_misses = torch.zeros(n, dtype=torch.long, device=device)
        self.sweep_unresolved = torch.zeros_like(self.sweep_misses)
        self.pending_sweep_age = torch.full_like(self.sweep_misses, -1)
        self._last_physics_frame = -1
        # Keep the 62.5 us PhysX integration and force-confirmed shuttle
        # contact, but sample robot jump/recovery metrics every 1 ms.  These
        # metrics are hysteretic over tens of milliseconds; recomputing them
        # at every native step was pure Python/tensor overhead and did not
        # change collision response or event accounting.
        self._jump_metric_stride = 16
        self.ball_episode_records = []

    def _setup_scene(self):
        super()._setup_scene()
        self.shuttle = RigidObject(self.cfg.shuttle)
        self.scene.rigid_objects["shuttle"] = self.shuttle
        self.scene.sensors["shuttle_contact"] = ContactSensor(self.cfg.shuttle_contact)
        material = sim_utils.RigidBodyMaterialCfg(
            restitution=PHYSICS["restitution"], static_friction=PHYSICS["friction"],
            dynamic_friction=PHYSICS["friction"], restitution_combine_mode="max")
        material.func("/World/ShuttleContactMaterial", material)
        stage = get_current_stage()
        bodies, colliders = [], []
        for i in range(self.num_envs):
            root = f"/World/envs/env_{i}"
            bodies += [root + "/Shuttle", root + "/Robot/Racket"]
            colliders += [root + "/Shuttle/geometry/mesh", root + "/Robot/Racket/Collider"]
        configure_ccd(stage, bodies, colliders, "/World/ShuttleContactMaterial")
        if self.cfg.include_net:
            net = sim_utils.CuboidCfg(size=(0.02, 2.0, self.cfg.net_height_m),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(restitution=0.1))
            for i in range(self.num_envs):
                path = f"/World/envs/env_{i}/Net"
                # InteractiveScene replication may already have copied the
                # source net into cloned environments.  Preserve one native
                # collider per environment instead of spawning duplicates.
                if stage.GetPrimAtPath(path).IsValid():
                    continue
                # Net is a child of the translated environment. The spawner's
                # translation is LOCAL; adding env_origins displaces it twice.
                if self.cfg.net_side_height_m > self.cfg.net_height_m:
                    from quadsmash.physics.net_geometry import spawn_profiled_net, PROFILE_VERSION
                    from isaaclab.sim.utils import bind_physics_material
                    if self.cfg.net_geometry_version != PROFILE_VERSION:
                        raise ValueError('Profiled geometry must have its own declared version')
                    spawn_profiled_net(stage, path, self.cfg.net_x_m, self.cfg.net_height_m,
                                       self.cfg.net_side_height_m, self.cfg.net_profile_segments)
                    material_path = path+'/geometry/material'
                    net.physics_material.func(material_path, net.physics_material)
                    bind_physics_material(path+'/geometry/mesh', material_path)
                else:
                    net.func(path, net,
                        translation=(self.cfg.net_x_m, 0., self.cfg.net_height_m/2))

    @torch.inference_mode()
    def _write_shuttle_reset(self, ids):
        state = self.shuttle.data.default_root_state[ids].clone()
        state[:, :3] = self.scene.env_origins[ids] + torch.tensor(self.cfg.shuttle_start, device=self.device)
        state[:, 7:10] = torch.tensor(self.cfg.shuttle_velocity, device=self.device)
        self.shuttle.write_root_state_to_sim(state, env_ids=ids)
        if self.contact_events is not None:
            self.previous_ball_position[ids] = state[:, :3]
            self.previous_ball_velocity[ids] = state[:, 7:10]
            self.previous_drag[ids] = 0.0

    @torch.inference_mode()
    def _reset_idx(self, ids):
        if self.contact_events is not None:
            complete = ids[self._episode_has_stepped[ids]]
            for i in complete.cpu().tolist():
                self.ball_episode_records.append({
                    "env_id": i, "contact": bool(self.contact_seen[i]),
                    "airborne_contact": bool(self.airborne_contact_seen[i]),
                    "body_contact": bool(self.body_contact_seen[i]), "contact_count": int(self.contact_events.count[i]),
                    "sweep_misses": int(self.sweep_misses[i]), "sweep_unresolved": int(self.sweep_unresolved[i]),
                    "contact_time_s": float(self.contact_time[i]), "contact_impulse_ns": float(self.contact_impulse[i]),
                    "ball_min_distance_m": float(self.ball_min_distance[i]),
                    "ball_min_distance_time_s": float(self.ball_min_distance_time[i]),
                    "ball_min_distance_ball_xyz": self.ball_min_distance_ball[i].detach().cpu().tolist(),
                    "ball_min_distance_tool_xyz": self.ball_min_distance_tool[i].detach().cpu().tolist(),
                    "outgoing_speed_mps": float(self.outgoing_speed[i]), "outgoing_vz_mps": float(self.outgoing_vz[i]),
                    "crossed_net": bool(self.crossed_net[i]), "landed_target": bool(self.landed_target[i]),
                    "recovery_success": bool(getattr(self, "jump_success", torch.zeros_like(self.contact_seen))[i])})
        super()._reset_idx(ids)
        self._write_shuttle_reset(ids)
        if self.contact_events is not None:
            self.contact_events.reset(ids)
            for name in ("contact_seen", "airborne_contact_seen", "body_contact_seen", "crossed_net", "landed_target", "landed"):
                getattr(self, name)[ids] = False
            for name in ("contact_impulse", "ball_clock", "outgoing_speed", "outgoing_vz", "_contact_bonus"):
                getattr(self, name)[ids] = 0.0
            self.contact_time[ids] = float("nan")
            self.ball_min_distance[ids] = float("inf")
            self.ball_min_distance_time[ids] = 0.0
            self.ball_min_distance_ball[ids] = 0.0
            self.ball_min_distance_tool[ids] = 0.0
            self._latest_racket_force[ids] = 0.0
            self.sweep_ready[ids] = False
            self.sweep_misses[ids] = 0
            self.sweep_unresolved[ids] = 0
            self.pending_sweep_age[ids] = -1

    def _pre_physics_step(self, actions):
        super()._pre_physics_step(actions)
        self._contact_bonus.zero_()

    def _apply_action(self):
        # Read the previous completed substep; DirectRLEnv updates scene data
        # after sim.step. The final substep is consumed in _get_dones below.
        frame = self._sim_step_counter - 1
        self._consume_physics_frame(frame)
        # Bypass JumpEnv's per-substep metric call while retaining the
        # CoupledEnv target write and foot-contact observation.
        CoupledEnv._apply_action(self)
        if frame == 0 or frame % self._jump_metric_stride == 0:
            self._consume_jump_frame(frame)
        force = drag_force(self.shuttle.data.root_lin_vel_w, self.cfg.shuttle_mass_kg,
                           self.cfg.drag_k_per_m, self.cfg.wind_m_s)
        self.previous_drag.copy_(force)
        self.shuttle.set_external_force_and_torque(force[:, None], torch.zeros_like(force[:, None]), is_global=True)

    def _consume_physics_frame(self, frame):
        if self.contact_events is None or frame <= self._last_physics_frame:
            return
        self._last_physics_frame = frame
        if frame == 0:
            return
        self._observe_physics_frame(frame)
        data, ball = self.robot.data, self.shuttle.data
        position, velocity = ball.root_pos_w, ball.root_lin_vel_w
        forces = self.shuttle_contact.data.force_matrix_w
        if forces.shape != (self.num_envs, 1, 2, 3):
            raise RuntimeError(f"Unexpected filtered contact shape: {forces.shape}")
        racket_force, ground_force = forces[:, 0, 0], forces[:, 0, 1]
        self._latest_racket_force.copy_(racket_force)
        gap = sphere_box_gap(position, data.body_pos_w[:, self.racket_body_id],
            data.body_quat_w[:, self.racket_body_id], PHYSICS["racket_half_extents_m"], self.cfg.shuttle_radius_m)
        new, touching = self.contact_events.update(racket_force, gap)
        # Independent swept geometry audits translation AND angular motion.
        # Contact response remains entirely in PhysX. A detected crossing that
        # never produces a force becomes a recorded hard numerical failure.
        tool_position = data.body_pos_w[:, self.racket_body_id]
        tool_quaternion = data.body_quat_w[:, self.racket_body_id]
        near_tool = (position - tool_position).square().sum(dim=-1) <= 0.45 ** 2
        swept = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        unresolved = torch.zeros_like(swept)
        if bool(near_tool.any()):
            # Avoid invoking the 256-iteration conservative advancement query
            # for the overwhelmingly common far-away shuttle frames.  The
            # query itself remains unchanged for every near-tool frame.
            hit, _, unresolved_near = swept_sphere_box(self.previous_ball_position[near_tool], position[near_tool],
                self.previous_tool_position[near_tool], tool_position[near_tool], self.previous_tool_quaternion[near_tool],
                tool_quaternion[near_tool], PHYSICS["racket_half_extents_m"], self.cfg.shuttle_radius_m)
            swept[near_tool] = hit
            unresolved[near_tool] = unresolved_near
        previous_gap = sphere_box_gap(self.previous_ball_position, self.previous_tool_position,
            self.previous_tool_quaternion, PHYSICS["racket_half_extents_m"], self.cfg.shuttle_radius_m)
        candidate = swept & self.sweep_ready & (previous_gap > 1e-5) & (~touching)
        pending = self.pending_sweep_age >= 0
        self.pending_sweep_age[pending] += 1
        self.pending_sweep_age[candidate & (~pending)] = 0
        self.pending_sweep_age[touching] = -1
        expired = self.pending_sweep_age >= 2
        self.sweep_misses += expired.long()
        self.pending_sweep_age[expired] = -1
        self.sweep_unresolved += (unresolved & self.sweep_ready).long()
        self.previous_tool_position.copy_(tool_position)
        self.previous_tool_quaternion.copy_(tool_quaternion)
        self.sweep_ready[:] = True
        self.ball_clock += self.physics_dt
        first = new & (~self.contact_seen)
        self.contact_time[first] = self.ball_clock[first]
        self._contact_bonus += first.float()
        self.contact_seen |= new
        feet_air = self.observed_airborne
        self.airborne_contact_seen |= new & feet_air
        gravity = torch.tensor(self.cfg.sim.gravity, device=self.device)
        impulse = self.cfg.shuttle_mass_kg * (velocity - self.previous_ball_velocity - gravity*self.physics_dt) - self.previous_drag*self.physics_dt
        self.contact_impulse += torch.where(touching, impulse.norm(dim=-1), 0.0)
        self.outgoing_speed[touching] = velocity[touching].norm(dim=-1)
        self.outgoing_vz[touching] = velocity[touching, 2]
        other_force = self.shuttle_contact.data.net_forces_w[:, 0] - racket_force - ground_force
        self.body_contact_seen |= other_force.norm(dim=-1) > PHYSICS["contact_force_threshold_n"]
        old_local, local = self.previous_ball_position - self.scene.env_origins, position - self.scene.env_origins
        crossing, _, point = segment_plane_crossing(old_local, local, 0, self.cfg.net_x_m)
        self.crossed_net |= crossing & self.contact_seen & (point[:, 2] - self.cfg.shuttle_radius_m > self.cfg.net_height_m)
        landing, _, point = segment_plane_crossing(old_local, local, 2, self.cfg.shuttle_radius_m, -1)
        ground_touch = ground_force.norm(dim=-1) > PHYSICS["contact_force_threshold_n"]
        contact_point = self.shuttle_contact.data.contact_pos_w[:, 0, 1] - self.scene.env_origins
        point = torch.where((~landing & ground_touch)[:, None], torch.nan_to_num(contact_point), point)
        just_landed = (landing | ground_touch) & (~self.landed)
        self.landed_target |= just_landed & self.contact_seen & self.crossed_net & (point[:, 0] >= self.cfg.target_x_min_m) & (point[:, 0] <= self.cfg.target_x_max_m) & (point[:, 1].abs() <= self.cfg.target_y_half_width_m)
        self.landed |= just_landed
        distance = (position - data.body_pos_w[:, self.racket_body_id]).norm(dim=-1)
        closer = distance < self.ball_min_distance
        self.ball_min_distance.copy_(torch.minimum(self.ball_min_distance, distance))
        self.ball_min_distance_time[closer] = self.ball_clock[closer]
        self.ball_min_distance_ball[closer] = position[closer] - self.scene.env_origins[closer]
        self.ball_min_distance_tool[closer] = data.body_pos_w[closer, self.racket_body_id] - self.scene.env_origins[closer]
        self.previous_ball_position.copy_(position)
        self.previous_ball_velocity.copy_(velocity)

    def _get_dones(self):
        self._consume_physics_frame(self._sim_step_counter)
        return super()._get_dones()

    def _get_rewards(self):
        return super()._get_rewards() + 0.25 * self._contact_bonus

    def _get_observations(self):
        # Every stage uses the same event/ball/history field layout.
        return super()._get_observations()
