"""Command-line options for the visualization tools."""

from pathlib import Path

from .visualizer import Visualizer


def _name_or_index(value):
    try:
        return int(value)
    except ValueError:
        return value


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz_path", type=Path, nargs="?")
    parser.add_argument("--interpolants", type=Path,
                        help="Interpolant archive: open a separate surface for every saved muscle")
    parser.add_argument("--resolution", type=int, default=60,
                        help="Surface samples per displayed axis (default: 60)")
    parser.add_argument("--path-only", action="store_true",
                        help="Show only the configured circular path in Simbody")
    parser.add_argument("--ocp-result", type=Path,
                        help="OCP result archive; overlays its trajectory on the dominance map")
    parser.add_argument("--model", "--model_path", dest="model_path", type=Path,
                        help="Matching .osim file (required for Simbody)")
    parser.add_argument("--field", choices=["workspace", *Visualizer._FIELDS], default=None)
    parser.add_argument("--space", choices=["cartesian", "joint"], default="cartesian")
    parser.add_argument("--coordinate-axes", type=int, nargs="+", metavar="INDEX",
                        help="Joint coordinate indices to plot (two or three).")
    parser.add_argument("--slice-coordinate", type=_name_or_index, help="Joint coordinate index or name to slice.")
    parser.add_argument("--muscle", type=_name_or_index, help="Muscle name or column index")
    parser.add_argument("--agonist-antagonist", action="store_true",
                        help="Overlay agonist/antagonist fatigue group means in the optimization view")
    parser.add_argument("--force", type=int, default=0, help="Flattened force index")
    parser.add_argument("--backend", choices=["matplotlib", "simbody"], default="matplotlib")
    parser.add_argument("--slice-axis", choices=["x", "y", "z"])
    parser.add_argument("--slice-value", type=float)
    parser.add_argument("--reference-marker", help="Slice/reference origin at this marker's default position")
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--max-points", type=int, help="Optional display-only subsampling")
    parser.add_argument("--radius", type=float, default=0.006, help="Simbody point radius in metres")
    parser.add_argument("--opacity", type=float, help="Point opacity in the range [0, 1]")
    parser.add_argument("--circle-r1", type=float, default=0.3, help="Circular path offset radius in metres")
    parser.add_argument("--circle-tz", type=float, default=0.0, help="Circular path translation along ground Z in metres")
    parser.add_argument("--circle-theta1", type=float, default=0.0, help="Circular path orientation in radians")
    parser.add_argument("--circle-height", type=float, default=0.6, help="Circular path height in metres")
    parser.add_argument("--circle-r2", type=float, default=0.01, help="Circular path radius in metres")
    args = parser.parse_args()
    if args.interpolants is not None:
        if args.path_only or args.ocp_result is not None or args.npz_path is not None:
            parser.error("--interpolants is a standalone view; omit other input modes")
        if args.coordinate_axes is not None and len(args.coordinate_axes) != 2:
            parser.error("The interpolant slice requires exactly two --coordinate-axes")
        if args.field not in (None, "fatigue", "muscle_force"):
            parser.error("Use --field fatigue or --field muscle_force for interpolants")
        Visualizer.show_interpolant_surface(
            args.interpolants, muscle=args.muscle, field=args.field or "fatigue",
            coordinate_axes=args.coordinate_axes or (0, 1), resolution=args.resolution,
        )
        return
    if args.path_only:
        if args.model_path is None:
            parser.error("--model is required with --path-only")
        if args.npz_path is not None or args.ocp_result is not None:
            parser.error("--path-only cannot be combined with a dataset or --ocp-result")
        from ..references import Circular1DOF
        viewer = Visualizer(model_path=args.model_path)
        viewer.show_circular_path(
            Circular1DOF(
                args.circle_r1, args.circle_tz, args.circle_theta1,
                args.circle_height, args.circle_r2,
            )
        )
        return
    if args.npz_path is None:
        parser.error("npz_path is required unless --path-only is used")
    viewer = Visualizer.from_npz(args.npz_path, model_path=args.model_path)
    if args.ocp_result is not None:
        if args.model_path is None:
            parser.error("--model is required with --ocp-result for the Simbody view")
        viewer.visualize_ocp_result(
            args.ocp_result,
            force_index=args.force,
            show_agonist_antagonist=args.agonist_antagonist,
            slice_axis=args.slice_axis,
            slice_value=args.slice_value,
            tolerance=args.tolerance,
            reference_marker=args.reference_marker,
            max_points=args.max_points,
            radius=args.radius,
            opacity=0.5 if args.opacity is None else args.opacity,
        )
    else:
        viewer.show(args.field or "dominance", muscle=args.muscle, force_index=args.force, backend=args.backend,
                    space=args.space, coordinate_axes=args.coordinate_axes,
                    slice_coordinate=args.slice_coordinate,
                    slice_axis=args.slice_axis, slice_value=args.slice_value, tolerance=args.tolerance,
                    reference_marker=args.reference_marker, max_points=args.max_points,
                    radius=args.radius,
                    opacity=1.0 if args.opacity is None else args.opacity)
    if args.backend == "simbody" or args.ocp_result is not None:
        input("Press Enter to close the viewer...")
