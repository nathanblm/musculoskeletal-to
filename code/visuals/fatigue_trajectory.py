"""Overlay an OCP trajectory on the current fatigue model's derivative gradient.

Place in code/visuals/fatigue_trajectory.py, then run:
    python -m code.visuals.fatigue_trajectory --model tsm_3DOF --muscle 0
    python -m code.visuals.fatigue_trajectory --model tsm_4DOF \
        --result results/tsm_4DOF/ocp/results/demo_3cc.npz --muscle TeresMinor

The model type is inferred from the saved state count. For 3CC the default
field is dMa/dt, because dMf/dt has no direct joint-space gradient at fixed
compartment states. Load only trusted NPZ result/dataset files.
"""
from pathlib import Path
from itertools import combinations

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.widgets import Slider
import numpy as np

from ..msk_data import MSKdataset, OCPResult, MODEL_CONFIG
from ..msk_fatiguemodel import Thresholded
from .fatigue_derivative import make_fatigue_model, make_field, switch_neighborhood


def show_fatigue_trajectory(model, dataset, result, *, muscle=0,
                            fatigue_model=None, field=None, force_index=0,
                            coordinate_axes=None, resolution=60, block=True,
                            show=True, save=None):
    """Return (figure, time_slider); accepts saved paths or existing data objects.

    By default show every unique joint pair, one row per pair.
    Pass coordinate_axes=(a, b) to show only one pair.
    At node k, plot the selected derivative at the saved muscle state,
    holding all hidden coordinates at q_k.
    Black arrows are unit negative spatial gradients (direction only).
    White/dark trails are coordinate projections; older points generally do not
    belong to the current slice. In 3D their heights are re-evaluated on the
    CURRENT slice, not their historical derivatives. Historical derivatives are
    shown separately in the time plot. No re-integration of fatigue is done.

    The dataset and model must match the ones used by the saved optimization.
    The result does not record their full provenance.
    """
    dataset = MSKdataset.from_npz(dataset) if isinstance(dataset, (str, Path)) else dataset
    result = OCPResult.from_npz(result) if isinstance(result, (str, Path)) else result
    result_muscles = tuple(result.muscle_names)
    if not result_muscles:
        raise ValueError('Saved result has no muscle names.')
    if isinstance(muscle, (int, np.integer)):
        if not 0 <= muscle < len(result_muscles):
            raise ValueError('Muscle index is outside the saved result muscle list.')
        muscle = result_muscles[int(muscle)]
    if muscle not in result_muscles:
        raise ValueError(f'{muscle!r} was not included in the saved result.')
    muscle_name = muscle
    names = tuple(dataset.info.joint_coordinate_names)
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
    count = states.shape[0] - 2 * nq
    inferred = ('thresholded' if count == len(result_muscles) else
                '3cc' if count in (2 * len(result_muscles), 3 * len(result_muscles)) else None)
    if inferred is None or states.shape[1] != time.size or indices.size != nq:
        raise ValueError('Saved result state shape does not match its joints and muscles.')
    recorded_model = getattr(result, 'fatigue_model', '')
    if recorded_model and recorded_model != inferred:
        raise ValueError(f'Saved model label {recorded_model!r} does not match the state layout ({inferred}).')
    if fatigue_model is not None and fatigue_model != inferred:
        raise ValueError(f'Saved result contains {inferred} states, not {fatigue_model}.')
    fatigue_model = inferred
    if field == 'active' and fatigue_model != '3cc':
        raise ValueError(
            'Saved result contains thresholded fatigue states, with no active '
            'compartment. Use --field fatigue or --field load; --field active '
            'requires a saved 3CC optimization result.'
        )
    fatigue = make_fatigue_model(model, dataset, muscle_name, fatigue_model, force_index)
    field = field or ('active' if fatigue_model == '3cc' else 'fatigue')
    evaluate, label = make_field(fatigue, field)
    interpolants = fatigue.interpolants
    q_path = states[indices, :]
    mi = result_muscles.index(muscle_name)
    if fatigue_model == 'thresholded':
        m_path = states[2 * nq + mi:2 * nq + mi + 1, :]
    else:
        m = len(result_muscles)
        if count == 2 * m:
            m_path = states[[2 * nq + mi, 2 * nq + m + mi], :]
        else:
            m_path = states[[2 * nq + 2 * m + mi, 2 * nq + mi], :]
    mf_path = m_path[0]
    if not np.isfinite(states).all() or not np.isfinite(time).all():
        raise ValueError('Result contains non-finite values.')
    grids = [np.asarray(g, float) for g in interpolants.axes_list]
    for i, g in enumerate(grids):
        if g.size < 2 or np.any(np.diff(g) <= 0):
            raise ValueError('Map axes must be strictly increasing with at least two points.')
        if np.any(q_path[i] < g[0] - 1e-8) or np.any(q_path[i] > g[-1] + 1e-8):
            raise ValueError(f'Trajectory leaves the interpolant domain for {names[i]}.')
    actual, grad, _ = evaluate.map(time.size)(q_path, m_path)
    actual, grad = np.asarray(actual).ravel(), np.asarray(grad)
    # Full-dimensional local alignment, using the optimizer's joint velocities.
    velocity = states[indices + 1, :]
    directional = np.sum(grad * velocity, axis=0)
    denominator = np.linalg.norm(grad, axis=0) * np.linalg.norm(velocity, axis=0)
    alignment = np.divide(-directional, denominator, out=np.full(time.size, np.nan),
                          where=denominator > 1e-12)
    slice_evaluate = evaluate.map(resolution ** 2)
    path_evaluate = evaluate.map(time.size)
    norm = Normalize(vmin=-1, vmax=1)
    fig = plt.figure(figsize=(15, max(9, 4.5 * len(pairs) + 4)))
    layout = fig.add_gridspec(len(pairs) + 1, 2,
                             height_ratios=[3] * len(pairs) + [1],
                             bottom=.18, top=.86, hspace=.55, wspace=.34)
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
        surface_axis.set(xlabel=names[a], ylabel=names[b], zlabel=label,
                         xlim=(x[0], x[-1]), ylim=(y[0], y[-1]),
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
    history.set(xlabel='Time [s]', ylabel=label)
    fatigue_axis.plot(time, mf_path, color='tab:red', label='Fatigued')
    if fatigue_model == '3cc':
        fatigue_axis.plot(time, m_path[1], color='tab:blue', label='Active')
        fatigue_axis.plot(time, 1 - m_path[0] - m_path[1], color='tab:green', label='Resting')
        fatigue_axis.legend(fontsize=8)
    fatigue_axis.set(xlabel='Time [s]', ylabel='Compartment fraction')
    cursors = [ax.axvline(time[0], color='black', ls='--') for ax in (history, fatigue_axis)]
    history_dot, = history.plot([], [], 'o', color='black')
    fatigue_dot, = fatigue_axis.plot([], [], 'o', color='black')
    slider = Slider(fig.add_axes([.2, .055, .62, .025]), 'Time [s]', time[0], time[-1],
                    valinit=time[0], valstep=time)

    def update(value):
        k = int(np.argmin(abs(time - value)))
        limit = 1e-12
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
            current_state = np.repeat(m_path[:, k:k+1], xx.size, axis=1)
            d, gradient, forces = slice_evaluate(points, current_state)
            d, forces = np.asarray(d).reshape(xx.shape), np.asarray(forces).reshape(xx.shape)
            gradient = np.asarray(gradient)
            limit = max(limit, float(np.max(np.abs(d))))
            for artist in artists:
                artist.remove()
            artists.clear()
            heat.set_array(d.ravel())
            if isinstance(fatigue, Thresholded):
                boundary = switch_neighborhood(forces >= fatigue.params.threshold[0])
            else:
                target = forces / fatigue.muscles.max_forces[0]
                ma = m_path[1, k]
                mr = 1 - m_path[0, k] - ma
                boundary = (switch_neighborhood(target > ma)
                            | switch_neighborhood(target - ma > mr))
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
            artists.append(surface_axis.plot_surface(xx, yy, d, cmap='coolwarm', norm=norm,
                                                     alpha=.72, linewidth=0, rcount=resolution, ccount=resolution))
            projected = np.repeat(q_path[:, k, None], time.size, axis=1)
            projected[a], projected[b] = q_path[a], q_path[b]
            heights = np.asarray(path_evaluate(
                projected, np.repeat(m_path[:, k:k+1], time.size, axis=1))[0]).ravel()
            trail.set_data(q_path[a, :k+1], q_path[b, :k+1])
            point.set_data([q_path[a, k]], [q_path[b, k]])
            projected_line.set_data_3d(q_path[a, :k+1], q_path[b, :k+1], heights[:k+1])
            current3d.set_data_3d([q_path[a, k]], [q_path[b, k]], [actual[k]])
            fixed = ', '.join(f'{name}={q_path[i,k]:.3f}'
                              for i, name in enumerate(names) if i not in (a, b))
            plane.set_title(f"Hidden joints: {fixed or 'none'}", fontsize=9)
        panels[0]['heat'].set_clim(-limit, limit)
        for panel in panels:
            panel['surface_axis'].set_zlim(-limit, limit)
        for cursor in cursors:
            cursor.set_xdata([time[k], time[k]])
        history_dot.set_data([time[k]], [actual[k]])
        fatigue_dot.set_data([time[k]], [mf_path[k]])
        cosine = f'{alignment[k]:+.2f}' if np.isfinite(alignment[k]) else 'undefined (zero speed/gradient)'
        fig.suptitle(f'{muscle_name} | {fatigue_model} | {field} | t={time[k]:.3f} s | '
                     f'Mf={mf_path[k]:.4f} | derivative={actual[k]:+.4g}\n'
                     f'Projected trajectory; arrows: negative spatial gradient; '
                     f'downhill alignment={cosine} (+1 downhill)')
        fig.canvas.draw_idle()

    slider.on_changed(update)
    fig._fatigue_trajectory = {'slider': slider, 'update': update,
                               'derivative': actual, 'downhill_alignment': alignment,
                               'directional_derivative': directional, 'panels': panels}
    update(time[0])
    if save is not None:
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=160, bbox_inches='tight')
    if show:
        plt.show(block=block)
    return fig, slider


def main():
    import argparse
    import opensim as osim

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='tsm_3DOF', choices=MODEL_CONFIG)
    parser.add_argument('--dataset')
    parser.add_argument('--result')
    parser.add_argument('--muscle', default='0')
    parser.add_argument('--fatigue-model', choices=('thresholded', '3cc'),
                        help='Load test_<model>.npz by default and validate its model type.')
    parser.add_argument('--field', choices=('fatigue', 'active', 'load'))
    parser.add_argument('--force-index', type=int, default=0)
    parser.add_argument('--coordinate-axes', nargs=2, type=int, default=None,
                        help='Show only this pair; default shows every joint pair.')
    parser.add_argument('--resolution', type=int, default=60)
    parser.add_argument('--save', type=Path)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    config = MODEL_CONFIG[args.model]
    muscle = int(args.muscle) if args.muscle.isdecimal() else args.muscle
    result_path = args.result or (
        str(Path(config['ocp_result_path']).with_name(f'test_{args.fatigue_model}.npz'))
        if args.fatigue_model else config['ocp_result_path']
    )
    show_fatigue_trajectory(
        osim.Model(config['model_path']), args.dataset or config['dataset_path'],
        result_path, muscle=muscle,
        fatigue_model=args.fatigue_model, field=args.field,
        force_index=args.force_index, coordinate_axes=args.coordinate_axes,
        resolution=args.resolution, show=not args.no_show, save=args.save,
    )


if __name__ == '__main__':
    main()
