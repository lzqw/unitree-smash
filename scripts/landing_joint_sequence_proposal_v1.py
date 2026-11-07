"""Offline joint-command sequence proposal, followed by nonlinear rejection.

All command knots are optimized together. The proposal freezes the loaded
M/h/J at the query state; it is intentionally NOT a reliable 0.4 s predictor.
Every candidate must be replayed with refreshed dynamics and original bounds.
Nothing in this module is installed in the native execution entrypoint.
"""
import copy
import time
import numpy as np
import osqp
import pinocchio as pin
from scipy import sparse
from scipy.linalg import lu_factor, lu_solve

from scripts.condensed_implicit_drive_v1 import condense_implicit_drive
from scripts.implicit_support_prediction_v1 import predict_implicit_support
from scripts.landing_joint_damper_mpc_v1 import joint_damper_constraints
from scripts.frame_point_kinematics_v1 import frame_point_kinematics


def finite_iterate_status(status):
    """Use installed OSQP constants; certificates are never position targets."""
    return status in {osqp.constant(name) for name in (
        'OSQP_SOLVED', 'OSQP_SOLVED_INACCURATE',
        'OSQP_MAX_ITER_REACHED', 'OSQP_TIME_LIMIT_REACHED')}


def knot_basis(ticks, knots):
    if type(ticks) is not int or type(knots) is not int or not 2 <= knots <= ticks <= 80:
        raise ValueError('Require 2..80 ticks and at most one knot per tick')
    locations = np.linspace(0., ticks-1., knots)
    return np.stack([np.interp(np.arange(ticks), locations, np.eye(knots)[i])
                     for i in range(knots)], axis=1)


def affine_sequence(mass, bias, selection, kp, kd, jac, jdot,
                    velocity, joint_position, basis, *, dt=.01, substeps=16, decay=.06):
    """Yield all microsteps as affine functions of all command knots.

The state propagated between controls is never reset to the query state.
The independent held-command kernel is used in tests, not in this derivation.
"""
    nv, na = selection.shape
    if (basis.ndim != 2 or not np.isfinite(basis).all() or
            np.any(basis < 0.) or not np.allclose(basis.sum(1), 1.) or
            type(substeps) is not int or substeps < 1 or not 0 < dt <= .02):
        raise ValueError('Finite convex command basis and physical microsteps required')
    nc = len(jac); width = basis.shape[1]*na; h = dt/substeps
    factor = lu_factor(np.block([
        [mass+h*(selection*(kd+h*kp))@selection.T, -h*jac.T],
        [jac, np.zeros((nc, nc))]]))
    maps = lu_solve(factor, np.c_[
        np.vstack((mass, (1-h/decay)*jac)),
        np.vstack((-h*selection*kp, np.zeros((nc, na)))),
        np.vstack((h*selection*kp, np.zeros((nc, na)))),
        np.r_[-h*bias, -h*jdot]])
    av, aq, au, b = maps[:, :nv], maps[:, nv:nv+na], maps[:, nv+na:nv+2*na], maps[:, -1]
    vm = np.zeros((nv, width)); vb = velocity.copy()
    qm = np.zeros((na, width)); qb = joint_position.copy()
    dm = np.zeros((nv, width)); db = np.zeros(nv)
    for tick, weights in enumerate(basis):
        um = np.kron(weights[None, :], np.eye(na))
        for micro in range(substeps):
            rm = av@vm+aq@qm+au@um; rb = av@vb+aq@qb+b
            vm, vb = rm[:nv], rb[:nv]
            qm = qm+h*selection.T@vm; qb = qb+h*selection.T@vb
            dm = dm+h*vm; db = db+h*vb
            yield dict(tick=tick, micro=micro, command_map=um,
                velocity_map=vm, velocity_bias=vb, joint_position_map=qm, joint_position_bias=qb,
                displacement_map=dm, displacement_bias=db,
                force_map=rm[nv:], force_bias=rb[nv:],
                torque_map=kp[:, None]*(um-qm)-kd[:, None]*(selection.T@vm),
                torque_bias=-kp*qb-kd*(selection.T@vb))


def propose_sequence(controller, state, reference, *, knots=9, time_limit_s=20.,
                     constraint_start='control_endpoints', nominal_profile=None,solver_backend='osqp',
                     differentiate_dynamics=False):
    """Return a bounded proposal, never an executable accepted MPC plan."""
    started = time.perf_counter()
    if not 0 < time_limit_s <= 60.: raise ValueError('Bounded offline solve required')
    if constraint_start not in ('empty', 'control_endpoints'):
        raise ValueError('Explicit constraint generation initialization required')
    if solver_backend not in ('osqp','active_set'):raise ValueError('Unknown QP backend')
    if differentiate_dynamics and nominal_profile is None:
        raise ValueError('Differentiated dynamics require a complete nominal trajectory')
    qp = controller.qp; model = qp.model; interface = qp.interface
    cfg = qp.settings; H = controller.settings.horizon_steps; dt = controller.settings.control_dt_s
    basis = knot_basis(H, knots); n = knots*18
    # snapshot arrays must survive subsequent Pinocchio calls on shared data.
    snap = copy.deepcopy(model.snapshot(state.q, state.v))
    cj, _ = qp.com_kinematics(state)
    cj = cj.copy()
    feet = np.flatnonzero(state.contacts)
    result = dict(native_execution=False, accepted=False, linear_proposal_solved=False,
        prediction='frozen-loaded-dynamics; fixed-observed-contact; unsaturated-PD',
        horizon_steps=H, control_dt_s=dt, knots=knots, decision_variables=n,
        constraint_start=constraint_start)
    result.update(solver_backend=solver_backend,
        per_call_hard_deadline=solver_backend=='osqp',
        external_process_guard_required=solver_backend=='active_set')
    if nominal_profile is not None:
        result['prediction']='time-varying-loaded-dynamics; nominal-contact-schedule; local-command-increments'
    if not len(feet):
        return dict(result, rejection='No observed support for this fixed-contact proposal')
    S = np.zeros((24, 18)); S[model.v_indices, np.arange(18)] = 1.
    jac = snap['foot_jacobian'][feet].reshape(-1, 24)
    jdot = (snap['foot_jacobian_dot'][feet]@state.v).reshape(-1)
    footids = [model.names.index(v) for v in model.feet]
    footpos = snap['body_positions_world'][np.asarray(footids)[feet]]
    center_xy = footpos[:, :2].mean(0)
    # Maintain loaded height; choose a support-centered horizontal endpoint.
    com_goal = snap['com_world'].copy(); com_goal[:2] = center_xy
    clearance_frames = frame_point_kinematics(model, state.q, state.v,
        ('FL_thigh', 'FR_thigh', 'RL_thigh', 'RR_thigh'))
    ground = float(np.median(footpos[:, 2]))
    constraints=[]; lows=[]; highs=[]; row_kinds=[]; initial_rows=set()
    P = np.eye(n)*1e-7; q = np.zeros(n)
    def bound(mat, bias, lo, hi, kind, *, seed=False):
        start=len(row_kinds); row_kinds.extend([kind]*len(bias))
        if constraint_start=='control_endpoints' and (seed or st['micro']==qp.substeps-1):
            initial_rows.update(range(start,start+len(bias)))
        constraints.append(mat); lows.append(np.broadcast_to(lo, len(bias))-bias)
        highs.append(np.broadcast_to(hi, len(bias))-bias)
    def task(mat, bias, target, weight):
        nonlocal P, q
        P += weight*mat.T@mat; q += weight*mat.T@(bias-target)
    last_um = None; last_bias=None
    stages = affine_sequence(snap['mass_matrix'], snap['nonlinear'], S,
        interface.kp, interface.kd, jac, jdot, state.v, state.joint_position,
        basis, dt=dt, substeps=qp.substeps, decay=cfg.contact_velocity_decay_s)
    if nominal_profile is not None:
        from scripts.landing_sequence_linearization_v1 import affine_trajectory_sequence
        stages=affine_trajectory_sequence(controller,state,basis,nominal_profile)
        if differentiate_dynamics:
            from scripts.landing_differentiated_sequence_v1 import affine_differentiated_sequence
            stages=affine_differentiated_sequence(controller,state,basis,nominal_profile)
            result['prediction']='differentiated-loaded-interval-dynamics; nominal-contact-schedule; local-command-increments'
    angles = np.arange(cfg.friction_facets)*2*np.pi/cfg.friction_facets
    cone = np.c_[np.cos(angles), np.sin(angles),
                 np.full(cfg.friction_facets, -interface.friction*np.cos(np.pi/cfg.friction_facets))]
    for st in stages:
        vm, vb = st['velocity_map'], st['velocity_bias']
        qm, qb = st['joint_position_map'], st['joint_position_bias']
        dm, db = st['displacement_map'], st['displacement_bias']
        bound(st['torque_map'], st['torque_bias'], -interface.effort, interface.effort,'motor_torque_nm')
        bound(S.T@vm, S.T@vb, -interface.velocity, interface.velocity,'joint_velocity_rad_s')
        bound(qm, qb, interface.limits[:, 0], interface.limits[:, 1],'joint_position_rad')
        gain = qp.damper_gain_s_inv
        bound(S.T@vm+gain*qm, S.T@vb+gain*qb,
              gain*interface.limits[:, 0], gain*interface.limits[:, 1],'joint_damper_rad_s')
        zmap = state.root_rotation_world[2]@dm[:3] if nominal_profile is None else dm[2]
        zbias = state.q[2]+(state.root_rotation_world[2]@db[:3] if nominal_profile is None else db[2])
        bound(zmap[None, :], np.array([zbias]), qp.minimum_root_height_m, np.inf,'root_height_m')
        for foot in range(len(st['force_bias'])//3):
            fm = st['force_map'][3*foot:3*foot+3]; fb = st['force_bias'][3*foot:3*foot+3]
            bound(fm[2:3], fb[2:3], 0., np.inf,'normal_force_n')
            bound(cone@fm, cone@fb, -np.inf, 0.,'friction_n')
        if st['micro'] == 0:
            um = st['command_map']; ub=st.get('command_bias',np.zeros(18))
            bound(um, ub, interface.limits[:, 0], interface.limits[:, 1],'command_range_rad',seed=True)
            delta = um if last_um is None else um-last_um
            offset = ub-(state.previous_command_target if last_um is None else last_bias)
            bound(delta[12:], offset[12:], -interface.arm_rate*dt, interface.arm_rate*dt,'arm_slew_rad',seed=True)
            task(um, ub, reference, .02/H)
            task(delta, offset, np.zeros(18), .2/H)
            last_um = um; last_bias=ub
        if st['micro'] == qp.substeps-1:
            terminal = st['tick'] == H-1
            w = 10. if terminal else 1./H
            if nominal_profile is None:
                position_jac=cj;position_bias=snap['com_world']+cj@db;velocity_jac=cj
            else:
                node=nominal_profile[st['tick']]
                position_jac=node['com_jac_world']
                position_bias=node['end_snapshot']['com_world']+position_jac@(db-node['displacement'])
                local_to_world=node['world_to_local'].T
                velocity_jac=position_jac@local_to_world
            task(position_jac@dm, position_bias, com_goal, 100.*w)
            task(velocity_jac@vm, velocity_jac@vb, np.zeros(3), 5.*w)
            task(vm[3:6], vb[3:6], np.zeros(3), 2.*w)
            # Relative orientation tangent towards original yaw, upright roll/pitch.
            yaw = np.arctan2(state.root_rotation_world[1, 0], state.root_rotation_world[0, 0])
            yaw_rotation = pin.rpy.rpyToMatrix(0., 0., yaw)
            upright_delta = pin.log3(state.root_rotation_world.T@yaw_rotation) if nominal_profile is None else pin.log3(yaw_rotation@state.root_rotation_world.T)
            task(dm[3:6], db[3:6], upright_delta, 10.*w)
            for frame in (clearance_frames if nominal_profile is None else node['frames']).values():
                # Link-origin reference only, as in the existing QP. Native
                # collision geometry remains an additional execution check.
                frame_jac=frame['jacobian'][2:3] if nominal_profile is None else frame['jacobian'][2:3]@node['world_to_local']
                frame_bias=np.array([frame['position'][2]])+frame_jac@(db if nominal_profile is None else db-node['displacement'])
                task(frame_jac@dm,frame_bias,
                     np.array([max(frame['position'][2], ground+cfg.nonfoot_clearance_m)]),
                     cfg.nonfoot_clearance_weight*w)
    if nominal_profile is not None:
        # Local model trust region; never a relaxation of hardware limits.
        bound(np.eye(n),np.zeros(n),-.1,.1,'local_increment_rad',seed=True)
        result['local_increment_bound_rad']=.1
    A = np.vstack(constraints); low = np.concatenate(lows); high = np.concatenate(highs)
    command_center = np.tile(reference, knots) if nominal_profile is None else np.zeros(n)
    scale = np.tile(.5*np.diff(interface.limits, axis=1).ravel(), knots)
    As = A*scale; rs = 1./np.maximum(np.max(abs(As), axis=1), 1e-12)
    Ps = P*scale[:, None]*scale[None, :]; qs = scale*(P@command_center+q)
    oscale = 1./max(1., np.max(abs(np.diag(Ps))))
    # Constraint generation changes only the numerical representation. Every
    # returned proposal must pass ALL original rows in physical units.
    chosen=initial_rows.copy(); warm=np.zeros(n); cut_log=[]; solve_start=time.perf_counter(); sol=None
    result['initial_constraint_rows']=len(initial_rows)
    if nominal_profile is not None:
        result['zero_increment_constraint_violation']=float(max(0.,np.max(low),np.max(-high)))
    for cutting_round in range(32):
        remaining=time_limit_s-(time.perf_counter()-solve_start)
        if remaining<=0:break
        indices=np.array(sorted(chosen),dtype=int)
        solver = osqp.OSQP()
        if solver_backend=='active_set':
            from scripts.dense_active_set_qp_v1 import DenseActiveSetQP
            solver=DenseActiveSetQP()
        solver.setup(P=sparse.csc_matrix(np.triu(Ps*oscale)), q=qs*oscale,
            A=sparse.csc_matrix(As[indices]*rs[indices,None]),
            l=(low[indices]-A[indices]@command_center)*rs[indices],
            u=(high[indices]-A[indices]@command_center)*rs[indices],
            eps_abs=1e-7, eps_rel=1e-7, max_iter=20000,
            polish=True,verbose=False,**({'time_limit':max(.001,remaining)} if solver_backend=='osqp' else {}))
        if warm is not None and hasattr(solver,'warm_start'):solver.warm_start(x=warm)
        sol=solver.solve()
        if sol.info.status_val!=1 or sol.x is None or not np.isfinite(sol.x).all():break
        warm=sol.x.copy(); trial=command_center+scale*warm; response=A@trial
        violations=np.maximum(low-response,response-high)
        cut_log.append(dict(round=cutting_round,rows=len(chosen),
            maximum_physical_violation=float(max(0.,np.max(violations)))))
        if np.max(violations)<=cfg.residual_tolerance:break
        candidates=np.flatnonzero(violations>cfg.residual_tolerance)
        candidates=np.array([i for i in candidates if int(i) not in chosen],dtype=int)
        if not len(candidates):break
        worst=candidates[np.argsort(violations[candidates]*rs[candidates])[-128:]]
        chosen.update(int(i) for i in worst)
    result['constraint_generation']=cut_log
    if sol is None:return dict(result,rejection='No sequence solve completed')
    result.update(solver_status=sol.info.status, iterations=sol.info.iter,
        constraint_rows=len(low), solver_runtime_s=sol.info.run_time,
        total_solver_wall_seconds=time.perf_counter()-solve_start,
        primal_residual=getattr(sol.info,'pri_res',getattr(solver,'last_diagnostics',{}).get('scaled_primal_residual')),
        dual_residual=getattr(sol.info,'dua_res',getattr(solver,'last_diagnostics',{}).get('stationarity')),
        wall_seconds=time.perf_counter()-started)
    # A finite iterate after timeout/iteration limit is diagnostic only.
    # Status numbers differ across OSQP releases. Infeasibility certificates
    # are not candidate commands and must not be decoded as joint angles.
    if finite_iterate_status(sol.info.status_val) and sol.x is not None and np.isfinite(sol.x).all():
        response=A@(command_center+scale*sol.x)
        violations=np.maximum(0.,np.maximum(low-response,response-high))
        kinds=np.asarray(row_kinds)
        result['all_row_violation_by_kind']={kind:float(np.max(violations[kinds==kind]))
                                            for kind in sorted(set(row_kinds))}
    if sol.info.status_val != 1 or sol.x is None or not np.isfinite(sol.x).all():
        return dict(result, rejection='Sequence QP not solved; no command authorized')
    x = command_center+scale*sol.x
    error = max(0., float(np.max(low-A@x)), float(np.max(A@x-high)))
    result['linear_constraint_violation'] = error
    if error > cfg.residual_tolerance:
        return dict(result, rejection='Independent unscaled sequence constraint check failed')
    commands=basis@x.reshape(knots,18)
    if nominal_profile is not None:commands+=np.stack([node['command'] for node in nominal_profile])
    result.update(linear_proposal_solved=True, commands=commands,
                  linear_terminal_com=position_bias+position_jac@dm@x)
    return result


def verify_sequence(controller, state, commands):
    """Refresh loaded dynamics every control tick; reject instead of clipping."""
    qp = controller.qp; model = qp.model; interface = qp.interface; cfg = qp.settings
    dt = controller.settings.control_dt_s; following = copy.deepcopy(state)
    saved_previous = copy.deepcopy(interface.previous_command)
    checks=[]; result=dict(accepted=False, native_execution=False, completed_stages=0)
    try:
        for index, command in enumerate(commands):
            interface.bind_previous_command(following.previous_command_target)
            lo, hi = interface.admitted_target_bounds(following.joint_position, dt)
            admission = max(0., float(np.max(lo-command)), float(np.max(command-hi)))
            if admission > cfg.residual_tolerance:
                return dict(result, rejection='Actual command admission differs', failed_stage=index)
            snap = copy.deepcopy(model.snapshot(following.q, following.v))
            active = np.flatnonzero(following.contacts)
            S = np.zeros((24, 18)); S[model.v_indices, np.arange(18)] = 1.
            stages = condense_implicit_drive(snap['mass_matrix'], snap['nonlinear'], S,
                interface.kp, interface.kd, snap['foot_jacobian'][active].reshape(-1, 24),
                (snap['foot_jacobian_dot'][active]@following.v).reshape(-1),
                following.v, following.joint_position, duration_s=dt, substeps=qp.substeps,
                contact_velocity_decay_s=cfg.contact_velocity_decay_s)
            mat, low, high = joint_damper_constraints(stages, S, interface.limits, qp.damper_gain_s_inv)
            damper = max(0., float(np.max(low-mat@command)), float(np.max(mat@command-high)))
            prediction = predict_implicit_support(model, interface, following, command,
                duration_s=dt, substeps=qp.substeps, contact_velocity_decay_s=cfg.contact_velocity_decay_s)
            diagnostics = prediction['diagnostics'].copy(); diagnostics.pop('maximum_active_set_iterations')
            diagnostics.update(command_admission=admission, joint_damper=damper)
            # Same microstep manifold root-height check as the interval QP.
            q = following.q.copy(); min_z = float(q[2])
            for stage in stages:
                q = pin.integrate(model.model, q, dt/qp.substeps*(stage['velocity_map']@command+stage['velocity_bias']))
                min_z = min(min_z, float(q[2]))
            diagnostics['root_height_violation_m'] = max(0., qp.minimum_root_height_m-min_z)
            checks.append(diagnostics)
            if max(diagnostics.values()) > cfg.residual_tolerance:
                return dict(result, rejection='Refreshed dynamics violates physical/envelope constraint',
                            failed_stage=index, checks=checks)
            following = prediction['state']; following.previous_command_target = command.copy()
            following, _ = controller._after_prediction(following)
            result['completed_stages'] = index+1
        risk = controller.risk(following)
        accepted = len(commands)==controller.settings.horizon_steps and controller._terminal_ok(following, risk)
        return dict(result, accepted=bool(accepted), terminal_risk=risk, checks=checks,
                    rejection=None if accepted else 'Original terminal recovery check failed')
    finally:
        interface.previous_command = saved_previous
