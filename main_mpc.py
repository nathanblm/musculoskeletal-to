import signal

import numpy as np
import opensim as osim
from code.msk_data import MODEL_CONFIG
from code.tasks import SphericalRegion, CircularPath2, CircularPath1
from code.msk_optimization import MusculoskeletalOCP
from code.msk_mcp import MusculoskeletalMPC
from code.msk_plant import O2Integrator
from code.msk_fatiguemodel import ThreeCompartment, Thresholded
from code.visuals.live_mpc import LiveMPCVisualizer


def main():
    model_name = "tsm_3DOF"
    config = MODEL_CONFIG[model_name]
    model = osim.Model(config["model_path"])


    muscles = [
        # arm26 models
        #"TRIlong",
        #"TRImed",
        #"TRIlat",
        #"BIClong",
        #"BICshort",
        #"BRA",

        # tsm models
        #"DeltoideusScapula_M", 
        "DeltoideusScapula_P",
        #"Subscapularis_S",
        "Subscapularis_I",
        #"Supraspinatus_A", 
        "Coracobrachialis", 
        "DeltoideusClavicle_A",
        #"DeltoideusScapula_P",
        #"DeltoideusScapula_M",
        #"PectoralisMajorClavicle_A",
        #"PectoralisMajorThorax_M",
        #"TeresMajor",
        "Infraspinatus_I",
        "Infraspinatus_S",
        "BIC_brevis",
        "TRIlong"
    ]
    horizon = 10
    timestep = 0.05
    steps_per_update = 1
    fatigue_model = "thresholded"  # "thresholded" or "3cc"
    force_index = 0

    task = SphericalRegion(
        center=np.array([-0.22686198, 0.39980641, 0.28600239]),
        radius=0.1,
    )

    #task = CircularPath2(
    #    r1=0.3,
    #    tz=0.2,
    #    h=0.6,
    #    r2=0.1,
    #)

    ocp = MusculoskeletalOCP.from_model(
        model=model,
        dataset=config["dataset_path"],
        force_index=force_index,
        muscles=muscles,
        fatigue_model_type={"thresholded": Thresholded, "3cc": ThreeCompartment}[fatigue_model],
        task=task,
        f_dynamics_func=None,
        i_dynamics_func=config["i_dynamics_func_path"],
        horizon=horizon,
        timestep=timestep,
    )

    plant = O2Integrator.from_model(
        model=osim.Model(config["model_path"]),
        dataset=config["dataset_path"],
        force_index=force_index,
        muscles=muscles,
        fatigue_model_type=fatigue_model,
        task=task,
        f_dynamics=None,
    )

    controller = MusculoskeletalMPC(ocp, plant, steps_per_update=steps_per_update)

    viewer = LiveMPCVisualizer(model, controller)
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print("Stopping after the current MPC update...")

    previous_handler = signal.signal(signal.SIGINT, request_stop)
    print("Running MPC. Press Ctrl+C to stop.")
    try:
        viewer.update()
        while not stop_requested:
            controller.step()
            viewer.update()
            print(f"t={plant.time:.2f} s, solve time={controller.solve_time:.3f} s")
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        viewer.close()

    return controller


if __name__ == "__main__":
    main()
