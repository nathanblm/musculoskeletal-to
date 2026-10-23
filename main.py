"""Run the dataset, OCP, RMR validation, and fatigue comparison pipeline."""

import os
from pathlib import Path

from code.msk_data import MODEL_CONFIG
from code.msk_datasetbuilder import build_msk_dataset, build_validation_dataset, dataset_demo
from code.msk_fatiguemodel import compare_muscle_state_trajectories
from code.msk_optimization import solve_ocp, visualize_ocp_result


ROOT = Path(__file__).resolve().parent

# Demo settings. Change these values to configure the pipeline.
MODEL_NAME = "tsm_4DOF"
N_SAMPLES = 18
FORCE_MAGNITUDES = [20.0]
FORCE_DIRECTIONS = [[0.0, -1.0, 0.0]]
FORCE_INDEX = 0
HORIZON = 400
TIMESTEP = 0.05
ACTIVATION_DYNAMICS = True
FILTER_SUCCESS = True
SHOW_DATASET_VISUALS = True
CYCLE_DOMINANCE_FORCES = False
SHOW_OCP_VISUAL = True
BLOCK_OCP_VISUAL = False
OCP_MUSCLES = ["TeresMinor", "Supraspinatus_A"] #"Supraspinatus_P",


def main():
    os.chdir(ROOT)

    dataset_path = Path(
        build_msk_dataset(
            MODEL_NAME,
            n_samples=N_SAMPLES,
            magnitudes=FORCE_MAGNITUDES,
            directions=FORCE_DIRECTIONS,
            tag="left",
            regular=True,
        )
    )
    MODEL_CONFIG[MODEL_NAME]["dataset_path"] = str(dataset_path)

    if SHOW_DATASET_VISUALS:
        dataset_demo(
            MODEL_NAME,
            str(dataset_path),
            endpoint_marker=MODEL_CONFIG[MODEL_NAME]["reference_marker"],
            cycle_dominance_forces=CYCLE_DOMINANCE_FORCES,
        )

    ocp_result_path = Path(MODEL_CONFIG[MODEL_NAME]["ocp_result_path"])
    solve_ocp(
        MODEL_NAME,
        force_index=FORCE_INDEX,
        horizon=HORIZON,
        timestep=TIMESTEP,
        tag=ocp_result_path.stem,
        muscles=OCP_MUSCLES,
    )

    if SHOW_OCP_VISUAL:
        visualize_ocp_result(
            MODEL_NAME,
            block=BLOCK_OCP_VISUAL,
        )

    build_validation_dataset(
        MODEL_NAME,
        str(dataset_path),
        str(ocp_result_path),
        force_index=FORCE_INDEX,
        activation_dynamics=ACTIVATION_DYNAMICS,
        filter_succes=FILTER_SUCCESS,
    )
    return compare_muscle_state_trajectories(
        MODEL_NAME,
        force_index=FORCE_INDEX,
    )


if __name__ == "__main__":
    main()
