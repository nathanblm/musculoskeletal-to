"""Inspect saved 3CC trajectories with the same joint-pair slices as fatigue_trajectory.

Run:
    python -m code.visuals.fatigue_trajectory_2
    python -m code.visuals.fatigue_trajectory_2 --field active --muscle Infraspinatus_S
    python -m code.visuals.fatigue_trajectory_2 --field load

The true dmf/dt = F*ma - R*mf has zero spatial gradient at fixed compartments.
Use --field active for dma/dt, or --field load for posture-dependent target load.
All fields hold current compartments and hidden joints fixed. No equilibrium
assumption is made. Trails are projections re-evaluated on the current slice.
Only load trusted NPZ results: the result/dataset loaders use pickle.
"""
from pathlib import Path
from itertools import combinations

import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.widgets import Slider
import numpy as np

from scipy.integrate import solve_ivp

from ..msk_data import Interpolants, MSKdataset, OCPResult, MODEL_CONFIG, MuscleStates
from ..msk_optimization_3cc import ThreeCCModel, ThreeCCParameters, ThreeCCResult


def show_fatigue_trajectory_3cc(dataset, interpolants, result, *, muscle=0,
                                coordinate_axes=None, resolution=60, block=True,
                                field="fatigue", initial_positions=None):
    """Return (slice figure, slider); also open a joint-motion comparison figure.

    Fields: fatigue (dmf/dt), active (dma/dt), load (clipped target load).
    A held-initial-pose replay is a counterfactual baseline, not the infeasible
    all-zero compartment trajectory supplied as IPOPT's initial guess.
    """
    dataset = MSKdataset.from_npz(dataset) if isinstance(dataset, (str, Path)) else dataset
    interpolants = Interpolants.from_npz(interpolants) if isinstance(interpolants, (str, Path)) else interpolants
    result = OCPResult.from_npz(result) if isinstance(result, (str, Path)) else result
    if not isinstance(result, ThreeCCResult):
        raise ValueError('Expected a saved ThreeCCResult from msk_optimization_3cc.')
    if field not in ('fatigue', 'active', 'load'):
        raise ValueError('field must be fatigue, active, or load.')
    names = tuple(interpolants.joint_coordinate_names)
    if result.joint_names and tuple(result.joint_names) != names:
        raise ValueError('Saved result and interpolant coordinate ordering differs.')
    if not np.allclose(result.endpoint_force, interpolants.force):
        raise ValueError('Saved result and interpolants use different endpoint forces.')
    muscles = tuple(interpolants.muscle_names)
    mi = muscles.index(muscle) if isinstance(muscle, str) else int(muscle)
    if not 0 <= mi < len(muscles):
        raise ValueError('Muscle index is outside the interpolant muscle list.')
    muscle_name = muscles[mi]
    if muscle_name not in result.muscle_names:
        raise ValueError(f'{muscle_name} was not included in this solution.')
    if names != tuple(dataset.info.joint_coordinate_names):
        raise ValueError('Dataset and interpolant coordinate ordering differs.')
    nq = len(names)
    if nq < 2:
        raise ValueError('At least two joint coordinates are needed for a surface.')
    pairs = (list(combinations(range(nq), 2)) if coordinate_axes is None else
             [tuple(names.index(i) if isinstance(i, str) else int(i)
                    for i in coordinate_axes)])
    for pair in pairs:
        if len(pair) != 2 or pair[0] == pair[1] or any(i < 0 or i >= nq for i in pair):
            raise ValueError('Select two distinct coordinate axes.')
    if resolution < 3:
        raise ValueError('resolution must be at least 3.')
    time = np.asarray(result.time, float).ravel()
    states = np.asarray(result.states, float)
    indices = np.asarray(result.pos_indices, int)
    if time.size < 2 or np.any(np.diff(time) <= 0):
        raise ValueError('Result needs at least two strictly increasing time points.')
    if states.shape != (2 * nq + len(result.muscle_names), time.size) or indices.size != nq:
        raise ValueError('Expected interleaved joint positions/velocities followed by muscle states.')
    q_path = states[indices, :]
    ri = tuple(result.muscle_names).index(muscle_name)
    m_path = states[2 * nq + ri, :]
    active_states = np.asarray(result.active_states, float)
    if active_states.shape != (len(result.muscle_names), time.size):
        raise ValueError('Saved active compartments do not match the trajectory.')
    if not np.isfinite(active_states).all():
        raise ValueError('Saved active compartments contain non-finite values.')
    active_path = active_states[ri]
    if not np.isfinite(states).all() or not np.isfinite(time).all():
        raise ValueError('Result contains non-finite values.')
    grids = [np.asarray(g, float) for g in interpolants.axes_list]
    for i, g in enumerate(grids):
        if g.size < 2 or np.any(np.diff(g) <= 0):
            raise ValueError('Map axes must be strictly increasing with at least two points.')
        if np.any(q_path[i] < g[0] - 1e-8) or np.any(q_path[i] > g[-1] + 1e-8):
            raise ValueError(f'Trajectory leaves the interpolant domain for {names[i]}.')
    muscle_indices = np.asarray([tuple(dataset.info.muscle_names).index(name)
                                 for name in result.muscle_names])
    muscle_states = MuscleStates(tuple(result.muscle_names), muscle_indices,
                                  states[2 * nq:, 0],
                                  np.asarray(dataset.info.muscle_max_forces)[muscle_indices])
    params = ThreeCCParameters(**result.parameters_3cc)
    model = ThreeCCModel(interpolants, muscle_states, params)
    q = ca.MX.sym('q', nq)
    mf, ma = ca.MX.sym('mf', len(result.muscle_names)), ca.MX.sym('ma', len(result.muscle_names))
    rhs = model.derivative(q, mf, ma)
    expressions = {'fatigue': rhs[ri], 'active': rhs[len(result.muscle_names) + ri],
                   'load': model.target_load(q)[ri]}
    mf_path = states[2 * nq:]
    expression = expressions[field]
    label = {'fatigue': 'dmf/dt [1/s]', 'active': 'dma/dt [1/s]',
             'load': 'Target load [fraction]'}[field]
    evaluate = ca.Function('trajectory_3cc_slice', [q, mf, ma],
                           [expression, ca.gradient(expression, q), model.target_load(q)[ri]])
    actual, grad, loads = evaluate.map(time.size)(q_path, mf_path, active_states)
    actual, grad, loads = np.asarray(actual).ravel(), np.asarray(grad), np.asarray(loads).ravel()
    saved_loads = np.asarray(result.target_loads)
    if saved_loads.shape == active_states.shape and not np.allclose(loads, saved_loads[ri], atol=1e-7, rtol=1e-6):
        raise ValueError('Force map/maximum forces differ from the saved 3CC solve (target loads do not match).')

    if initial_positions is None:
        initial_positions = getattr(result, 'initial_joint_positions', None)
    if initial_positions is None or np.size(initial_positions) == 0:
        raise ValueError('Supply initial_positions or use the CLI to reconstruct the initial IK guess.')
    initial_positions = np.asarray(initial_positions, float).reshape(nq)
    if not np.isfinite(initial_positions).all():
        raise ValueError('Initial positions must be finite.')
    z = ca.MX.sym('compartments', 2 * len(result.muscle_names))
    held_rhs = ca.Function('held_ik_rhs', [z], [model.derivative(
        initial_positions, z[:len(result.muscle_names)], z[len(result.muscle_names):])])
    replay = solve_ivp(lambda t, value: np.asarray(held_rhs(value)).ravel(),
                       (time[0], time[-1]), np.r_[mf_path[:, 0], active_states[:, 0]],
                       t_eval=time, max_step=float(np.min(np.diff(time))), rtol=1e-7, atol=1e-10)
    if not replay.success:
        raise RuntimeError(f'Held-initial-pose replay failed: {replay.message}')
    held_fatigue = replay.y[:len(result.muscle_names)]
    movement = np.max(np.abs(q_path - initial_positions[:, None]), axis=1)
    replay_objective = 0.1 * (time[1] - time[0]) * np.sum(held_fatigue**2)
    print('Max joint deviation from initial IK guess [rad]:')
    for name, delta in zip(names, movement):
        print(f'  {name}: {delta:.6g}')
    print(f'Saved objective: {result.objective:.6g}; held-IK replay objective: {replay_objective:.6g}')
    print('Held-IK replay keeps the initial pose fixed; the original zero-fatigue solver guess is not a dynamically feasible reference.')
    if field == 'fatigue':
        print('3CC dmf/dt has no direct q dependence: flat slices and zero spatial arrows are expected. Try --field active or --field load.')
    # Full-dimensional local alignment, using the optimizer's joint velocities.
    velocity = states[indices + 1, :]
    directional = np.sum(grad * velocity, axis=0)
    denominator = np.linalg.norm(grad, axis=0) * np.linalg.norm(velocity, axis=0)
    alignment = np.divide(-directional, denominator, out=np.full(time.size, np.nan),
                          where=denominator > 1e-12)
    slice_evaluate = evaluate.map(resolution ** 2)
    path_evaluate = evaluate.map(time.size)
    limit = {'fatigue': max(params.fatigue_rate, params.recovery_rate),
             'active': max(params.development, params.relaxation) + params.fatigue_rate,
             'load': 1.0}[field]
    limit = max(limit, float(np.max(np.abs(actual))), 1e-12)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    fig = plt.figure(figsize=(15, 4.5 * len(pairs) + 3))
    layout = fig.add_gridspec(len(pairs) + 1, 2,
                             height_ratios=[3] * len(pairs) + [1],
                             bottom=.09, top=.91, hspace=.45, wspace=.30)
    panels = []
    for row, (a, b) in enumerate(pairs):
        x = np.linspace(grids[a][0], grids[a][-1], resolution)
        y = np.linspace(grids[b][0], grids[b][-1], resolution)
        xx, yy = np.meshgrid(x, y)
        plane = fig.add_subplot(layout[row, 0])
        surface_axis = fig.add_subplot(layout[row, 1], projection='3d')
        heat = plane.pcolormesh(xx, yy, np.zeros_like(xx), shading='auto', cmap='coolwarm', norm=norm)
        plane.plot(q_path[a], q_path[b], color='.3', ls=':', alpha=.6, label='Full path (projection)')
        plane.plot(initial_positions[a], initial_positions[b], marker='X', color='gold',
                   mec='black', ms=9, ls='', label='Initial IK guess')
        trail, = plane.plot([], [], color='white', lw=3, label='Traversed (projection)')
        point, = plane.plot([], [], 'o', color='lime', mec='black', ms=9, label='Current configuration')
        plane.set(xlabel=names[a], ylabel=names[b], xlim=(x[0], x[-1]), ylim=(y[0], y[-1]))
        plane.legend(fontsize=8, loc='upper right')
        surface_axis.set(xlabel=names[a], ylabel=names[b], zlabel=label,
                         xlim=(x[0], x[-1]), ylim=(y[0], y[-1]), zlim=(-limit, limit),
                         title=f'{names[a]} / {names[b]}: current slice')
        projected_line, = surface_axis.plot([], [], [], color='black', lw=2)
        current3d, = surface_axis.plot([], [], [], 'o', color='lime', mec='black', ms=8)
        panels.append(dict(a=a, b=b, xx=xx, yy=yy, plane=plane,
                           surface_axis=surface_axis, heat=heat, trail=trail,
                           point=point, projected_line=projected_line,
                           current3d=current3d, artists=[]))
    fig.colorbar(panels[0]['heat'], ax=[p['plane'] for p in panels],
                 label=label, fraction=.025, pad=.03)
    history = fig.add_subplot(layout[-1, 0])
    fatigue_axis = fig.add_subplot(layout[-1, 1])
    history.plot(time, actual, color='tab:blue')
    history.axhline(0, color='.5', lw=.7)
    history.set(xlabel='Time [s]', ylabel=f'Historical {label}')
    fatigue_axis.plot(time, m_path, color='tab:red', label='Optimized mf')
    fatigue_axis.plot(time, held_fatigue[ri], '--', color='black', label='Held initial IK pose (replay)')
    if result.val_states.shape == states.shape:
        fatigue_axis.plot(time, result.val_states[2 * nq + ri], ':', color='tab:orange', label='Optimized path RK45')
    fatigue_axis.legend(fontsize=8)
    fatigue_axis.set(xlabel='Time [s]', ylabel=f'{muscle_name}: fatigued mf')
    cursors = [ax.axvline(time[0], color='black', ls='--') for ax in (history, fatigue_axis)]
    history_dot, = history.plot([], [], 'o', color='black')
    fatigue_dot, = fatigue_axis.plot([], [], 'o', color='black')
    slider = Slider(fig.add_axes([.2, .055, .62, .025]), 'Time [s]', time[0], time[-1],
                    valinit=time[0], valstep=time)

    def update(value):
        k = int(np.argmin(abs(time - value)))
        for panel in panels:
            a, b = panel['a'], panel['b']
            xx, yy = panel['xx'], panel['yy']
            plane, surface_axis = panel['plane'], panel['surface_axis']
            heat, trail, point = panel['heat'], panel['trail'], panel['point']
            projected_line, current3d = panel['projected_line'], panel['current3d']
            artists = panel['artists']
            points = np.empty((nq, xx.size))
            points[:] = q_path[:, k, None]
            points[a], points[b] = xx.ravel(), yy.ravel()
            d, gradient, _ = slice_evaluate(
                points, np.repeat(mf_path[:, k:k+1], xx.size, axis=1),
                np.repeat(active_states[:, k:k+1], xx.size, axis=1))
            d = np.asarray(d).reshape(xx.shape)
            gradient = np.asarray(gradient)
            for artist in artists:
                artist.remove()
            artists.clear()
            heat.set_array(d.ravel())
            step = max(1, resolution // 13)
            gx = -gradient[a].reshape(xx.shape)
            gy = -gradient[b].reshape(xx.shape)
            length = np.hypot(gx, gy)
            valid = length > 1e-12
            ux = np.divide(gx, length, out=np.zeros_like(gx), where=valid)
            uy = np.divide(gy, length, out=np.zeros_like(gy), where=valid)
            sl = np.s_[::step, ::step]
            artists.append(plane.quiver(xx[sl], yy[sl], ux[sl], uy[sl],
                                       angles='xy', scale_units='inches', scale=9, color='black', alpha=.65))
            artists.append(surface_axis.plot_surface(xx, yy, d, cmap='coolwarm', norm=norm,
                                                     alpha=.72, linewidth=0, rcount=resolution, ccount=resolution))
            projected = np.repeat(q_path[:, k, None], time.size, axis=1)
            projected[a], projected[b] = q_path[a], q_path[b]
            heights = np.asarray(path_evaluate(
                projected, np.repeat(mf_path[:, k:k+1], time.size, axis=1),
                np.repeat(active_states[:, k:k+1], time.size, axis=1))[0]).ravel()
            trail.set_data(q_path[a, :k+1], q_path[b, :k+1])
            point.set_data([q_path[a, k]], [q_path[b, k]])
            projected_line.set_data_3d(q_path[a, :k+1], q_path[b, :k+1], heights[:k+1])
            current3d.set_data_3d([q_path[a, k]], [q_path[b, k]], [actual[k]])
            fixed = ', '.join(f'{name}={q_path[i,k]:.3f}'
                              for i, name in enumerate(names) if i not in (a, b))
            plane.set_title(f"Hidden joints: {fixed or 'none'}", fontsize=9)
        for cursor in cursors:
            cursor.set_xdata([time[k], time[k]])
        history_dot.set_data([time[k]], [actual[k]])
        fatigue_dot.set_data([time[k]], [m_path[k]])
        cosine = f'{alignment[k]:+.2f}' if np.isfinite(alignment[k]) else 'undefined (zero speed/gradient)'
        fig.suptitle(f'3CC {muscle_name} | t={time[k]:.3f} s | mf={m_path[k]:.4f} | ma={active_path[k]:.4f} | {label}={actual[k]:+.4g}\n'
                     f'Current compartments and hidden joints fixed; field={field}\n'
                     f'Arrows: negative spatial gradient | Full-space downhill alignment={cosine} (+1 downhill, -1 uphill)')
        fig.canvas.draw_idle()

    slider.on_changed(update)
    fig._fatigue_trajectory = {'slider': slider, 'update': update,
                               'derivative': actual, 'downhill_alignment': alignment,
                               'directional_derivative': directional, 'panels': panels,
                               'initial_positions': initial_positions, 'held_fatigue': held_fatigue,
                               'held_objective': replay_objective, 'field': field}
    motion_fig, motion_axes = plt.subplots(nq, 1, squeeze=False, sharex=True,
                                           figsize=(10, max(5, 2.3 * nq)))
    for i, name in enumerate(names):
        ax = motion_axes[i, 0]
        ax.plot(time, q_path[i], label='Optimized joint position')
        ax.axhline(initial_positions[i], color='black', ls='--', label='Initial IK guess')
        ax.set(ylabel=f'{name} [rad]', title=f'Max deviation from guess: {movement[i]:.4g} rad')
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    motion_axes[-1, 0].set_xlabel('Time [s]')
    motion_fig.suptitle('3CC movement relative to the solver initial pose')
    motion_fig.tight_layout()
    fig._fatigue_trajectory['motion_figure'] = motion_fig
    update(time[0])
    plt.show(block=block)
    return fig, slider


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='tsm_3DOF', choices=MODEL_CONFIG)
    parser.add_argument('--dataset')
    parser.add_argument('--interpolants')
    parser.add_argument('--result')
    parser.add_argument('--muscle', default='0')
    parser.add_argument('--coordinate-axes', nargs=2, type=int, default=None,
                        help='Show only this pair; default shows every joint pair.')
    parser.add_argument('--resolution', type=int, default=60)
    parser.add_argument('--field', choices=('fatigue', 'active', 'load'), default='fatigue',
                        help='fatigue: true dmf/dt (flat at fixed states); active: dma/dt; load: target load')
    args = parser.parse_args()
    config = MODEL_CONFIG[args.model]
    muscle = int(args.muscle) if args.muscle.isdecimal() else args.muscle
    result_path = args.result or Path('results') / args.model / 'ocp_3cc' / 'results' / 'test.npz'
    result = OCPResult.from_npz(result_path)
    if not isinstance(result, ThreeCCResult):
        raise ValueError('Expected a saved 3CC result.')
    initial_positions = getattr(result, 'initial_joint_positions', None)
    if initial_positions is None or np.size(initial_positions) == 0:
        # Older results did not save IPOPT's initial pose. Reconstruct it using
        # precisely the baseline initializer: IK of the initial task center.
        import opensim as osim
        from ..sk_kinematics import Kinematics
        kin = Kinematics(osim.Model(config['model_path']))
        initial_positions, _ = kin.inverse_kinematics(np.asarray(result.task)[:, 0])
        print('Reconstructed initial IK pose from current model and saved initial task center.')
    show_fatigue_trajectory_3cc(args.dataset or config['dataset_path'],
                                args.interpolants or config['interpolants_path'], result,
                                muscle=muscle, coordinate_axes=args.coordinate_axes,
                                resolution=args.resolution, field=args.field,
                                initial_positions=initial_positions)


if __name__ == '__main__':
    main()