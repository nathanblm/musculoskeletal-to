# Pipeline demo

From the repository root, activate your project environment and run:

```bash
python main.py
```

The script builds and visualizes the muscle fatigue map, optimizes and visualizes a trajectory with the OCP, generates RMR validation data, then compares the map based and RMR based muscle state trajectories. Close the dataset visualization and press Enter when prompted to continue.

Configure the demo at the top of [main.py](main.py):

- `MODEL_NAME`: model key, such as `arm26_3DOF`.
- `N_SAMPLES`: number of sampled joint configurations used to build the map.
- `FORCE_MAGNITUDES` and `FORCE_DIRECTIONS`: endpoint forces included in the map; `FORCE_INDEX` selects one for the OCP and validation.
- `HORIZON` and `TIMESTEP`: OCP horizon length and time step.
- `OCP_TAG`: shared name for the optimization and validation outputs; the demo uses `demo_3cc`.
- `ACTIVATION_DYNAMICS` and `FILTER_SUCCESS`: options for generating RMR validation data.
- `SHOW_DATASET_VISUALS` and `SHOW_OCP_VISUAL`: enable or skip the two visualization steps.
- `BLOCK_OCP_VISUAL`: set to `True` to wait for the OCP plot window to close before continuing; defaults to `False`.

Increase `N_SAMPLES` for a denser map. Model files and generated dynamics must be available at the paths configured in the pipeline modules.

### Cost surfaces with a Simbody trajectory reference

`main_ocp.py` opens one Matplotlib figure containing three 3D cost surfaces
(XY, XZ, YZ) and one Simbody window showing the arm and its trajectory when
`SHOW_COST_VISUALS = True`. Both use Y-up camera orientations. In each surface,
the omitted spatial coordinate is replaced by cost change ΔJ: XY has cost along
the Z direction, XZ along Y, and YZ along X. Axis labels identify the cost axis;
it is not a physical endpoint coordinate. Surface colors use the same symmetric
scale in all three panels at each time step: blue means lower cost and red higher
cost. Black arrows point downhill. Cyan stars mark sampled interior slice minima.

The full path is grey, the traversed path orange, and the current position green
in every plot. Paths on the surfaces are projected into the selected Cartesian
slice and their heights are re-evaluated on the **current** cost surface; they
are not historical costs. Simbody shows the actual unprojected motion. Invalid
inverse-kinematics samples leave gaps. Mouse rotation of the Matplotlib surfaces
is preserved when stepping through time.

Use the **Previous** and **Next** buttons below the surfaces, or **[** and **]**
in either window. All plots share one motion-time index. Arrow keys also work in
Matplotlib. **Q** or closing the Matplotlib figure closes both windows. Times
clamp at the first and last nodes. Each fatigue-model comparison opens after the
previous one is closed. Separate older trajectory/fatigue views can be enabled
with `SHOW_TRAJECTORY_VISUAL` and `SHOW_GRADIENT_VISUALS`.

To generate and inspect an archive without running the full pipeline:

```bash
python -m code.msk_optimization --model tsm_3DOF --fatigue-model 3cc --cost-landscape --no-show --no-save-motion
python -m code.visuals.cost_surfaces results/tsm_3DOF/ocp/results/test_3cc_cost.npz --model tsm_3DOF
```

For the short example already generated:

```bash
python -m code.visuals.cost_surfaces results/tsm_3DOF/ocp/results/cost_preview_3cc_cost.npz --model tsm_3DOF
```

`--span 0.04` sets the slice half-width in metres; `--resolution 35` increases
sampling (use an odd number >= 5). For an image without desktop windows, use
`--no-simbody --no-show --save cost.png`. Reopen cost archives without solving
again; plain result archives lack the objective and constraint graph. Load only
trusted cost archives. Use the OpenSim environment for the desktop viewer.

Each surface changes only the selected node's posture in the actual OCP
objective. Fatigue, controls and other nodes stay fixed; fatigue is not replayed.
Inverse kinematics follows the local branch from that posture. Lower-cost
postures can violate dynamics or task constraints. These partial gradients and
slice minima cannot establish a constrained local optimum; the time controls
traverse motion time rather than solver iterations.
