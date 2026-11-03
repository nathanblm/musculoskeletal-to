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

## Position-feedback MPC

Run the matched-plant experiment in the same environment as the OCP demo:

```bash
python main_mpc.py
```

This uses the existing dataset configured in `MODEL_CONFIG`; generate it with the
original pipeline first if it is missing. Settings at the top of `main_mpc.py`
select the model, muscles, force condition, prediction horizon, and run duration.
`FATIGUE_MODEL` selects `"thresholded"` (the default) or `"3cc"`; the same MPC
controller supports both state layouts. The current prediction covers 1.2
seconds (`25` nodes at `0.05 s`).

`DURATION = None` runs continuously until Ctrl+C. The current update is allowed
to finish, then the complete executed trajectory and timing data are saved
without waiting for the plot windows. Set `DURATION` to seconds for a finite run.
Output filenames append the selected fatigue model to `TAG`, for example
`demo_thresholded.npz` or `demo_3cc.npz`.

The thresholded MPC uses `THRESHOLD_TRANSITION_WIDTH = 0.01`, a smooth transition
with a width of 1% of each muscle's maximum force around its loading threshold.
The discontinuous hard switch caused IPOPT to reach its iteration limit in the
demo. Both prediction and fatigue estimation use the smoothed equations. Set
the width to `0` to use the original hard switch. Offline OCPs still use the hard
switch by default. The selected width is recorded in the companion timing file.

`SHOW_SIMBODY = True` opens the OpenSim/Simbody window during the MPC run. The
arm follows the simulated plant after each feedback update. Red shows the
executed endpoint trail, orange shows the current endpoint prediction, grey
shows the task path, and the blue arrow shows the fixed interaction force.
The caption reports simulation time and the most recent solve time. Rendering
follows solver progress rather than wall-clock simulation time. Press Enter
after the run to close the viewer, or set `BLOCK_SIMBODY = False` to finish
without waiting. Set `SHOW_SIMBODY = False` for runs without a display.

`SHOW_FATIGUE = True` opens a separate live Matplotlib window. Solid lines show
the accumulated muscle-state estimate; dashed lines show the latest prediction
on the same absolute simulation-time axis. The three-compartment model shows
fatigued, active, and resting fractions in separate panels. Close the plot after
the run to continue, or set `BLOCK_FATIGUE = False` to finish without waiting.
The plots update between solves; the optimizer runs synchronously.
`DISPLAY_HISTORY` limits both live histories to the most recent 20 seconds by
default. Full state and control histories remain in memory and are saved when
the run stops.

The prediction is solved again at every feedback update. Only its latest path
is drawn, replacing the previous prediction. Overlapping paths are expected for
the matched plant with an unchanged task and force condition. Each console line
reports the RMS endpoint prediction revision in millimeters, comparing overlapping
future times with the previous plan shifted by the executed intervals. The first
update reports `nan` because there is no previous plan. Simbody shows the arm at
the current prediction's start time, with orange markers at the predicted start
and end and an update counter in the caption.

The plant is an acceleration-controlled double integrator, matching the current
OCP demo. Only joint positions are fed back. The controller estimates velocity
from successive positions and the applied acceleration, and integrates fatigue
along the measured posture history. Muscle forces come from the posture-dependent
dataset map at a fixed interaction-force condition (`FORCE_INDEX`). Fatigue is a
model estimate; it is not measured or corrected from muscle feedback.

`MusculoskeletalOCP` builds its constraints once and accepts a complete initial
state on each `solve()`. Calling `solve()` without arguments still runs the
offline OCP, now with initial joint positions and velocities constrained to the
initial trajectory state. MPC carries fatigue, task parameters, and task rates
between horizons and shifts the previous prediction as its next initial guess.
`MOVING_TASK_REGION = False` holds the starting sphere center fixed in the MPC
demo. Set it to `True` to let the controller optimize the center as before.
`SphericalRegion` still defaults to a moving center for existing offline OCPs.
The fixed region matters for load sharing: moving the region can let the
controller find a posture where both selected muscles recover together.

`COST_OPTIONS` in `main_mpc.py` controls the stage objective:

```text
w_fatigue      * sum(fatigue**2)
+ w_fatigue_rate * sum(max(fatigue_derivative, 0)**2)
+ w_overlap     * sum(load[i] * load[j] for each selected muscle pair)
+ sum(w_control * control**2)
```

The MPC starts with weights `1.0`, `0.1`, `1.0`, and `1e-5`, respectively.
Set any weight to zero to disable that term. `muscle_pairs` contains names from
`MUSCLES`; update it when changing the selected muscles. The demo penalizes overlap
for TeresMajor/Infraspinatus_S, selected by screening both dataset resolutions
for two reachable postures with opposing loading patterns under the new cost.
Pair selection is explicit;
physical proximity is not used to define pairs.

`load` is nonnegative muscle force divided by maximum isometric force. For the
thresholded model it is also multiplied by the dynamics' loading indicator,
including its configured smooth transition. The rate penalty only uses positive
derivatives of the fatigued fractions: recovery and the 3CC active-compartment
derivative do not incur that penalty. The control penalty applies to acceleration
in the matched-plant demo. The prediction also retains the existing final-node
fatigue penalty, with weight `dt * w_fatigue`.

The same objective supports offline OCPs via `solve_ocp(..., cost_options=...)`
or `MusculoskeletalOCP.from_model(..., cost_options=...)`. Without cost options,
offline runs retain the original fatigue-only objective. The added costs favor
resting one muscle while another works, but do not prescribe a switching cycle.

The selected pair has dataset postures loading TeresMajor/Infraspinatus_S at
approximately 30%/8% and 0.5%/13% of maximum force. A sampled straight joint-space
connection stays inside the fixed 10 cm sphere. A twelve-second thresholded MPC
screening showed repeated exchanges between the loading patterns; this is a
simulation observation, not a prescribed two-point reference. The endpoint and
joint angles remain free inside the region. Dataset rankings, positions, and
simulation traces are saved under
`results/tsm_3DOF/mpc/position_pair_selection/`.

`STEPS_PER_UPDATE` sets the number of prediction intervals between feedback
updates. The returned acceleration is held for that entire interval, and the
corresponding first prediction controls are constrained equal. The velocity
estimate assumes this acceleration-controlled plant; revisit it when adding
actuator dynamics or noisy measurements. Optional `DISTURBANCE_TIME` and
`DISTURBANCE` settings introduce a joint-position jump to exercise feedback.

Executed trajectories are saved as `OCPResult` archives under
`results/<model>/mpc/results/`. Companion timing archives contain each solve time
and the feedback interval, plus cost weights, selected pairs, and integrated cost
components (`cost_names` and `cost_values`). The executed objective integrates all
four stage costs on the executed states and applied controls, using the feedback
interval. It does not include a terminal penalty or sum overlapping prediction
objectives. The cost components are also printed when the run finishes.
`val_states` is empty because no independent validation trajectory is generated.
The script runs as a simulation without enforcing wall-clock deadlines. Compare
solve times with the feedback interval before attempting real-time use; a failed
solve stops the simulation rather than applying a new control.

Focused checks:

```bash
python -m unittest discover -s tests -p test_msk_mpc.py
```
