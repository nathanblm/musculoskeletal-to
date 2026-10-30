"""Inspect an OCP trajectory on its evolving thresholded fatigue derivative map.

Place in code/visuals/fatigue_trajectory.py, then run:
    python -m code.visuals.fatigue_trajectory --model tsm_3DOF --muscle 0
    python -m code.visuals.fatigue_trajectory --result results/run.npz --muscle MUSCLE

Only load trusted dataset/result NPZs: their existing loaders use pickle.
No OpenSim or FatigueModel construction is needed. Units match the saved map.
"""
from pathlib import Path
from itertools import combinations

import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.widgets import Slider
import numpy as np

from ..msk_data import Interpolants, MSKdataset, OCPResult, MODEL_CONFIG, FatigueParameters


def show_fatigue_trajectory(dataset, interpolants, result, *, muscle=0,
                            coordinate_axes=None, resolution=60, block=True):
    """Return (figure, time_slider); accepts saved paths or existing data objects.

    By default show every unique joint pair, one row per pair.
    Pass coordinate_axes=(a, b) to show only one pair.
    At node k, plot D(q, m_k), holding all hidden coordinates at q_k.
    Black arrows are unit negative spatial gradients (direction only).
    White/dark trails are coordinate projections; older points generally do not
    belong to the current slice. In 3D their heights are re-evaluated on the
    CURRENT slice, not their historical derivatives. Historical derivatives are
    shown separately in the time plot. No re-integration of fatigue is done.

    The supplied interpolants and fatigue parameters must be those used by the
    solve. The result format does not record that provenance automatically.
    """
    dataset = MSKdataset.from_npz(dataset) if isinstance(dataset, (str, Path)) else dataset
    interpolants = Interpolants.from_npz(interpolants) if isinstance(interpolants, (str, Path)) else interpolants
    result = OCPResult.from_npz(result) if isinstance(result, (str, Path)) else result
    names = tuple(interpolants.joint_coordinate_names)
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
    m_path = states[2 * nq + tuple(result.muscle_names).index(muscle_name), :]
    if not np.isfinite(states).all() or not np.isfinite(time).all():
        raise ValueError('Result contains non-finite values.')
    grids = [np.asarray(g, float) for g in interpolants.axes_list]
    for i, g in enumerate(grids):
        if g.size < 2 or np.any(np.diff(g) <= 0):
            raise ValueError('Map axes must be strictly increasing with at least two points.')
        if np.any(q_path[i] < g[0] - 1e-8) or np.any(q_path[i] > g[-1] + 1e-8):
            raise ValueError(f'Trajectory leaves the interpolant domain for {names[i]}.')
    pi = tuple(dataset.info.muscle_names).index(muscle_name)
    params = dataset.fatigue_params
    # Match FatigueModel: the active-force threshold comes from the current
    # FatigueParameters defaults, not a stale threshold serialized in the dataset.
    threshold = float(FatigueParameters.from_max_forces(
        dataset.info.muscle_max_forces
    ).threshold[pi])
    recovery = float(params.recovery / params.capacity[pi])
    q, m = ca.MX.sym('q', nq), ca.MX.sym('m')
    rate = interpolants.interpolant('fatigue')(q)[mi]
    force = interpolants.interpolant('muscle_force')(q)[mi]
    derivative = ca.if_else(force >= threshold, (1 - m) * rate, -m * recovery)
    evaluate = ca.Function('trajectory_derivative', [q, m],
                           [derivative, ca.gradient(derivative, q), force])
    actual, grad, _ = evaluate.map(time.size)(q_path, m_path[None, :])
    actual, grad = np.asarray(actual).ravel(), np.asarray(grad)
    # Full-dimensional local alignment, using the optimizer's joint velocities.
    velocity = states[indices + 1, :]
    directional = np.sum(grad * velocity, axis=0)
    denominator = np.linalg.norm(grad, axis=0) * np.linalg.norm(velocity, axis=0)
    alignment = np.divide(-directional, denominator, out=np.full(time.size, np.nan),
                          where=denominator > 1e-12)
    slice_evaluate = evaluate.map(resolution ** 2)
    path_evaluate = evaluate.map(time.size)
    all_rates = np.asarray(interpolants.fatigue_rate_grid).reshape(-1, len(muscles))[:, mi]
    limit = max(float(np.max(np.abs(all_rates))), abs(recovery), 1e-12)
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
        trail, = plane.plot([], [], color='white', lw=3, label='Traversed (projection)')
        point, = plane.plot([], [], 'o', color='lime', mec='black', ms=9, label='Current configuration')
        plane.set(xlabel=names[a], ylabel=names[b], xlim=(x[0], x[-1]), ylim=(y[0], y[-1]))
        plane.legend(fontsize=8, loc='upper right')
        surface_axis.set(xlabel=names[a], ylabel=names[b], zlabel='dm/dt [1/s]',
                         xlim=(x[0], x[-1]), ylim=(y[0], y[-1]), zlim=(-limit, limit),
                         title=f'{names[a]} / {names[b]}: current slice')
        projected_line, = surface_axis.plot([], [], [], color='black', lw=2)
        current3d, = surface_axis.plot([], [], [], 'o', color='lime', mec='black', ms=8)
        panels.append(dict(a=a, b=b, xx=xx, yy=yy, plane=plane,
                           surface_axis=surface_axis, heat=heat, trail=trail,
                           point=point, projected_line=projected_line,
                           current3d=current3d, artists=[]))
    fig.colorbar(panels[0]['heat'], ax=[p['plane'] for p in panels],
                 label='dm/dt [1/s]', fraction=.025, pad=.03)
    history = fig.add_subplot(layout[-1, 0])
    fatigue_axis = fig.add_subplot(layout[-1, 1])
    history.plot(time, actual, color='tab:blue')
    history.axhline(0, color='.5', lw=.7)
    history.set(xlabel='Time [s]', ylabel='Historical dm/dt [1/s]')
    fatigue_axis.plot(time, m_path, color='tab:red')
    fatigue_axis.set(xlabel='Time [s]', ylabel=f'{muscle_name}: fatigue m')
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
            d, gradient, forces = slice_evaluate(points, np.full((1, xx.size), m_path[k]))
            d, forces = np.asarray(d).reshape(xx.shape), np.asarray(forces).reshape(xx.shape)
            gradient = np.asarray(gradient)
            for artist in artists:
                artist.remove()
            artists.clear()
            heat.set_array(d.ravel())
            # No arrows at the switching boundary: the derivative is discontinuous there.
            active = forces >= threshold
            boundary = np.zeros_like(active)
            boundary[1:] |= active[1:] != active[:-1]
            boundary[:-1] |= active[1:] != active[:-1]
            boundary[:, 1:] |= active[:, 1:] != active[:, :-1]
            boundary[:, :-1] |= active[:, 1:] != active[:, :-1]
            step = max(1, resolution // 13)
            gx = -gradient[a].reshape(xx.shape)
            gy = -gradient[b].reshape(xx.shape)
            length = np.hypot(gx, gy)
            valid = (length > 1e-12) & ~boundary
            ux = np.divide(gx, length, out=np.zeros_like(gx), where=valid)
            uy = np.divide(gy, length, out=np.zeros_like(gy), where=valid)
            sl = np.s_[::step, ::step]
            artists.append(plane.quiver(xx[sl], yy[sl], ux[sl], uy[sl],
                                       angles='xy', scale_units='inches', scale=9, color='black', alpha=.65))
            if np.min(forces) < threshold < np.max(forces):
                artists.append(plane.contour(xx, yy, forces, levels=[threshold], colors='black', linewidths=1))
            artists.append(surface_axis.plot_surface(xx, yy, d, cmap='coolwarm', norm=norm,
                                                     alpha=.72, linewidth=0, rcount=resolution, ccount=resolution))
            projected = np.repeat(q_path[:, k, None], time.size, axis=1)
            projected[a], projected[b] = q_path[a], q_path[b]
            heights = np.asarray(path_evaluate(projected, np.full((1, time.size), m_path[k]))[0]).ravel()
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
        fig.suptitle(f'{muscle_name} | t={time[k]:.3f} s | m={m_path[k]:.4f} | dm/dt={actual[k]:+.4g}\n'
                     f'All slices use current fatigue and current hidden joints\n'
                     f'Arrows: negative spatial gradient | Full-space downhill alignment={cosine} (+1 downhill, -1 uphill)')
        fig.canvas.draw_idle()

    slider.on_changed(update)
    fig._fatigue_trajectory = {'slider': slider, 'update': update,
                               'derivative': actual, 'downhill_alignment': alignment,
                               'directional_derivative': directional, 'panels': panels}
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
    args = parser.parse_args()
    config = MODEL_CONFIG[args.model]
    muscle = int(args.muscle) if args.muscle.isdecimal() else args.muscle
    show_fatigue_trajectory(args.dataset or config['dataset_path'],
                            args.interpolants or config['interpolants_path'],
                            args.result or config['ocp_result_path'], muscle=muscle,
                            coordinate_axes=args.coordinate_axes, resolution=args.resolution)


if __name__ == '__main__':
    main()