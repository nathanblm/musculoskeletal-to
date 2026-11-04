"""Run position-and-velocity-feedback MPC with the dataset force condition held fixed."""

import os
import signal
from collections import deque
from pathlib import Path

import numpy as np
import opensim as osim

from code.msk_data import MODEL_CONFIG, OCPResult
from code.msk_fatiguemodel import ThreeCompartment, Thresholded
from code.msk_mpc import MusculoskeletalMPC
from code.msk_optimization import MusculoskeletalOCP
from code.msk_plant import JointPlant
from code.msk_rmr_feedback import RMRFatigueObserver
from code.tasks import SphericalRegion
from code.visuals.simbody_renderer import SimbodyRenderer
from code.visuals.live_fatigue import LiveFatiguePlot


ROOT = Path(__file__).resolve().parent
MODEL_NAME = "tsm_3DOF"
FORCE_INDEX = 0
HORIZON = 25
TIMESTEP = 0.05
RMR_TIMESTEP = 0.01  # Maximum interval for executed-motion force/fatigue updates.
STEPS_PER_UPDATE = 1
FATIGUE_MODEL = "3cc"  # "thresholded" or "3cc"
THRESHOLD_TRANSITION_WIDTH = 0.01  # Fraction of maximum force; 0 uses the hard switch.
DURATION = None  # Run until Ctrl+C; set seconds for a finite experiment.
DISPLAY_HISTORY = 20.0  # Seconds of executed history in the live windows.
TAG = "demo"  # The selected fatigue model is appended to the filename.
MUSCLES = [
    "Coracobrachialis",
    "TRIlong",
    "Infraspinatus_S",
]
SHOW_FATIGUE = True
BLOCK_FATIGUE = True  # Keep the fatigue plot open after the simulation.
SHOW_SIMBODY = True
BLOCK_SIMBODY = True  # Keep the final frame open until Enter is pressed.
# Optional instantaneous position disturbance, in radians.
DISTURBANCE_TIME = None
DISTURBANCE = np.array([0.01, 0.0, 0.0])


def main():
    os.chdir(ROOT)
    config = MODEL_CONFIG[MODEL_NAME]
    model = osim.Model(config["model_path"])
    # As in the OCP demo, the sphere center is optimized and moves with the task.
    task = SphericalRegion(
        center=np.array([-0.22686198, 0.39980641, 0.28600239]), radius=0.1,
    )
    ocp = MusculoskeletalOCP.from_model(
        model, None, config["i_dynamics_func_path"], config["dataset_path"],
        task, HORIZON, TIMESTEP, muscles=MUSCLES,
        force_index=FORCE_INDEX,
        fatigue_model_type={"thresholded": Thresholded, "3cc": ThreeCompartment}[FATIGUE_MODEL],
        fatigue_model_options=({"transition_width": THRESHOLD_TRANSITION_WIDTH}
                               if FATIGUE_MODEL == "thresholded" else None),
    )
    ocp.conf.s_opts = dict(ocp.conf.s_opts, print_level=0)
    ocp.opti.solver("ipopt", ocp.conf.p_opts, ocp.conf.s_opts)
    controller = MusculoskeletalMPC(ocp, steps_per_update=STEPS_PER_UPDATE)
    plant = JointPlant(ocp.initial_state[:ocp.joint_states.size:2],
                       ocp.initial_state[1:ocp.joint_states.size:2])
    # RMR modifies its model, so give it a separate OpenSim instance.
    observer = RMRFatigueObserver.from_model(
        osim.Model(config["model_path"]), ocp.fatigue_model,
        controller.state[ocp.joint_states.size:],
    )
    if not np.isfinite(RMR_TIMESTEP) or RMR_TIMESTEP <= 0:
        raise ValueError("RMR_TIMESTEP must be positive and finite")
    dt = controller.sample_time
    rmr_substeps = max(1, int(np.ceil(dt / RMR_TIMESTEP)))
    rmr_dt = dt / rmr_substeps
    n_steps = None if DURATION is None else round(DURATION / dt)
    display_steps = max(1, round(DISPLAY_HISTORY / dt))
    states, controls, task_points, solve_times = [], [], [], []
    positions = plant.positions.copy()
    velocities = plant.velocities.copy()
    renderer = SimbodyRenderer(model) if SHOW_SIMBODY else None
    fatigue_plot = LiveFatiguePlot(ocp.fatigue_model) if SHOW_FATIGUE else None
    endpoint_trail = deque(maxlen=display_steps + 1)
    if renderer is not None:
        renderer.set_joint_positions(positions)
        endpoint_trail.append(renderer.marker_position("endpoint"))
        renderer.show(np.empty((0, 3)), np.empty((0, 3)), "MPC | initial posture",
                      force=ocp.fatigue_model.interpolants.force)
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print("Stopping after the current MPC update; saving the executed trajectory.")

    previous_handler = signal.signal(signal.SIGINT, request_stop)
    k = 0
    print(f"Fatigue model: {FATIGUE_MODEL}. Press Ctrl+C to stop and save.")
    try:
        while not stop_requested and (n_steps is None or k < n_steps):
            if DISTURBANCE_TIME is not None and k == round(DISTURBANCE_TIME / dt):
                plant.positions += DISTURBANCE
                positions = plant.positions.copy()
            acceleration = controller.step(positions, velocities, observer.state)
            states.append(controller.state.copy())
            controls.append(acceleration)
            task_points.append(np.asarray(task.evaluate(
                ocp.task_parameters_solution[:, 0])).ravel())
            solve_times.append(controller.solve_time)
            print(f"update={k + 1} | t={k * dt:.2f} s | solve={controller.solve_time:.3f} s"
                  f" | prediction revision={1000 * controller.prediction_revision:.3f} mm RMS")
            if fatigue_plot is not None:
                fatigue_plot.update(
                    np.arange(max(0, k + 1 - display_steps), k + 1) * dt,
                    np.column_stack(states[-display_steps:])[ocp.joint_states.size:],
                    k * dt + controller.result.time,
                    controller.result.states[ocp.joint_states.size:],
                )
            if renderer is not None:
                renderer.set_joint_positions(positions, k * dt)
                endpoint_trail.append(renderer.marker_position("endpoint"))
                renderer.show(
                    np.empty((0, 3)), np.empty((0, 3)),
                    f"MPC update {k + 1} | t={k * dt:.2f} s | solve={controller.solve_time:.3f} s"
                    " | red: executed | orange: prediction | grey: task",
                    force=ocp.fatigue_model.interpolants.force,
                    trajectory=np.asarray(endpoint_trail),
                    reference=controller.result.task.T,
                    prediction=controller.result.endpoint_positions.T,
                )
            for _ in range(rmr_substeps):
                observer.advance(positions, velocities, acceleration, rmr_dt)
                positions, velocities = plant.step(acceleration, rmr_dt)
            k += 1
    finally:
        signal.signal(signal.SIGINT, previous_handler)

    n_steps = k
    if n_steps == 0:
        print("Stopped before any control interval was executed.")
        return None

    controller.update_state(positions, velocities, observer.state)
    states.append(controller.state.copy())
    task_points.append(np.asarray(task.evaluate(controller.task_parameters)).ravel())
    states = np.column_stack(states)
    if fatigue_plot is not None:
        fatigue_plot.update(
            np.arange(max(0, n_steps + 1 - display_steps), n_steps + 1) * dt,
            states[ocp.joint_states.size:, -display_steps:],
            (n_steps - 1) * dt + controller.result.time,
            controller.result.states[ocp.joint_states.size:],
        )
    if renderer is not None:
        renderer.set_joint_positions(positions, n_steps * dt)
        endpoint_trail.append(renderer.marker_position("endpoint"))
        renderer.show(
            np.empty((0, 3)), np.empty((0, 3)), f"MPC complete | t={n_steps * dt:.2f} s",
            force=ocp.fatigue_model.interpolants.force,
            trajectory=np.asarray(endpoint_trail), reference=controller.result.task.T,
            prediction=controller.result.endpoint_positions.T,
        )
    position_indices = np.arange(0, ocp.joint_states.size, 2)
    result = OCPResult(
        time=np.arange(n_steps + 1) * dt,
        states=states, val_states=np.empty((0, 0)), pos_indices=position_indices,
        controls=np.column_stack(controls),
        endpoint_positions=np.asarray(ocp.kinematics.endpoint_function(states[position_indices])),
        task=np.column_stack(task_points),
        objective=float(0.1 * dt * np.sum(np.asarray(
            ocp.fatigue_model.fatigue(states[ocp.joint_states.size:, :-1]))**2)),
        tag=f"{TAG}_{FATIGUE_MODEL}", muscle_names=ocp.fatigue_model.muscles.names,
        fatigue_model=FATIGUE_MODEL,
    )
    output = ROOT / "results" / MODEL_NAME / "mpc" / "results" / f"{result.tag}.npz"
    result.save_npz(output)
    np.savez_compressed(output.with_name(f"{result.tag}_timing.npz"),
                        solve_times=solve_times, sample_time=dt, rmr_timestep=rmr_dt,
                        fatigue_model=FATIGUE_MODEL,
                        threshold_transition_width=getattr(ocp.fatigue_model, "transition_width", 0.0))
    print(f"Saved executed trajectory: {output}")
    if fatigue_plot is not None:
        import matplotlib.pyplot as plt
        plt.show(block=BLOCK_FATIGUE and not stop_requested)
    if renderer is not None and BLOCK_SIMBODY and not stop_requested:
        input("Press Enter to close the Simbody viewer...")
    return result


if __name__ == "__main__":
    main()
