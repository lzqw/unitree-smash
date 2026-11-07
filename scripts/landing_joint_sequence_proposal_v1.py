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


def propose_sequence(controller, state, reference, *, knots=9, time_limit_s=20.):
    """Return a bounded proposal, never an executable accepted MPC plan."""
    started = time.perf_counter()
    if not 0 < time_limit_s <= 60.: raise ValueError('Bounded offline solve required')
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
        horizon_steps=H, control_dt_s=dt, knots=knots, decision_variables=n)
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
    constraints=[]; lows=[]; highs=[]
    P = np.eye(n)*1e-7; q = np.zeros(n)
    def bound(mat, bias, lo, hi):
        constraints.append(mat); lows.append(np.broadcast_to(lo, len(bias))-bias)
        highs.append(np.broadcast_to(hi, len(bias))-bias)
    def task(mat, bias, target, weight):
        nonlocal P, q
        P += weight*mat.T@mat; q += weight*mat.T@(bias-target)
    last_um = None
    stages = affine_sequence(snap['mass_matrix'], snap['nonlinear'], S,
        interface.kp, interface.kd, jac, jdot, state.v, state.joint_position,
        basis, dt=dt, substeps=qp.substeps, decay=cfg.contact_velocity_decay_s)
    angles = np.arange(cfg.friction_facets)*2*np.pi/cfg.friction_facets
    cone = np.c_[np.cos(angles), np.sin(angles),
                 np.full(cfg.friction_facets, -interface.friction*np.cos(np.pi/cfg.friction_facets))]
    for st in stages:
        vm, vb = st['velocity_map'], st['velocity_bias']
        qm, qb = st['joint_position_map'], st['joint_position_bias']
        dm, db = st['displacement_map'], st['displacement_bias']
        bound(st['torque_map'], st['torque_bias'], -interface.effort, interface.effort)
        bound(S.T@vm, S.T@vb, -interface.velocity, interface.velocity)
        bound(qm, qb, interface.limits[:, 0], interface.limits[:, 1])
        gain = qp.damper_gain_s_inv
        bound(S.T@vm+gain*qm, S.T@vb+gain*qb,
              gain*interface.limits[:, 0], gain*interface.limits[:, 1])
        zmap = state.root_rotation_world[2]@dm[:3]
        zbias = state.q[2]+state.root_rotation_world[2]@db[:3]
        bound(zmap[None, :], np.array([zbias]), qp.minimum_root_height_m, np.inf)
        for foot in range(len(feet)):
            fm = st['force_map'][3*foot:3*foot+3]; fb = st['force_bias'][3*foot:3*foot+3]
            bound(fm[2:3], fb[2:3], 0., np.inf)
            bound(cone@fm, cone@fb, -np.inf, 0.)
        if st['micro'] == 0:
            um = st['command_map']
            bound(um, np.zeros(18), interface.limits[:, 0], interface.limits[:, 1])
            delta = um if last_um is None else um-last_um
            offset = -state.previous_command_target if last_um is None else np.zeros(18)
            bound(delta[12:], offset[12:], -interface.arm_rate*dt, interface.arm_rate*dt)
            task(um, np.zeros(18), reference, .02/H)
            task(delta, offset, np.zeros(18), .2/H)
            last_um = um
        if st['micro'] == qp.substeps-1:
            terminal = st['tick'] == H-1
            w = 10. if terminal else 1./H
            task(cj@dm, snap['com_world']+cj@db, com_goal, 100.*w)
            task(cj@vm, cj@vb, np.zeros(3), 5.*w)
            task(vm[3:6], vb[3:6], np.zeros(3), 2.*w)
            # Relative orientation tangent towards original yaw, upright roll/pitch.
            yaw = np.arctan2(state.root_rotation_world[1, 0], state.root_rotation_world[0, 0])
            yaw_rotation = pin.rpy.rpyToMatrix(0., 0., yaw)
            upright_delta = pin.log3(state.root_rotation_world.T@yaw_rotation)
            task(dm[3:6], db[3:6], upright_delta, 10.*w)
            for frame in clearance_frames.values():
                # Link-origin reference only, as in the existing QP. Native
                # collision geometry remains an additional execution check.
                task(frame['jacobian'][2:3]@dm,
                     np.array([frame['position'][2]])+frame['jacobian'][2:3]@db,
                     np.array([max(frame['position'][2], ground+cfg.nonfoot_clearance_m)]),
                     cfg.nonfoot_clearance_weight*w)
    A = np.vstack(constraints); low = np.concatenate(lows); high = np.concatenate(highs)
    command_center = np.tile(reference, knots)
    scale = np.tile(.5*np.diff(interface.limits, axis=1).ravel(), knots)
    As = A*scale; rs = 1./np.maximum(np.max(abs(As), axis=1), 1e-12)
    Ps = P*scale[:, None]*scale[None, :]; qs = scale*(P@command_center+q)
    oscale = 1./max(1., np.max(abs(np.diag(Ps))))
    # Constraint generation changes only the numerical representation. Every
    # returned proposal must pass ALL original rows in physical units.
    chosen=set(); warm=None; cut_log=[]; solve_start=time.perf_counter(); sol=None
    for cutting_round in range(32):
        remaining=time_limit_s-(time.perf_counter()-solve_start)
        if remaining<=0:break
        indices=np.array(sorted(chosen),dtype=int)
        solver = osqp.OSQP()
        solver.setup(P=sparse.csc_matrix(np.triu(Ps*oscale)), q=qs*oscale,
            A=sparse.csc_matrix(As[indices]*rs[indices,None]),
            l=(low[indices]-A[indices]@command_center)*rs[indices],
            u=(high[indices]-A[indices]@command_center)*rs[indices],
            eps_abs=1e-7, eps_rel=1e-7, max_iter=20000,
            time_limit=max(.001,remaining),polish=True,verbose=False)
        if warm is not None:solver.warm_start(x=warm)
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
        primal_residual=sol.info.pri_res,dual_residual=sol.info.dua_res,
        wall_seconds=time.perf_counter()-started)
    if sol.info.status_val != 1 or sol.x is None or not np.isfinite(sol.x).all():
        return dict(result, rejection='Sequence QP not solved; no command authorized')
    x = command_center+scale*sol.x
    error = max(0., float(np.max(low-A@x)), float(np.max(A@x-high)))
    result['linear_constraint_violation'] = error
    if error > cfg.residual_tolerance:
        return dict(result, rejection='Independent unscaled sequence constraint check failed')
    result.update(linear_proposal_solved=True, commands=basis@x.reshape(knots, 18),
                  linear_terminal_com=snap['com_world']+cj@(dm@x+db))
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
