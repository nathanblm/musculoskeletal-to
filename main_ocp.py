"""Run the dataset, OCP, RMR validation, and fatigue comparison pipeline."""

import os
from pathlib import Path

from code.msk_data import MODEL_CONFIG
from code.msk_datasetbuilder import build_msk_dataset, build_validation_dataset
from code.msk_fatiguemodel import compare_muscle_state_trajectories
from code.msk_optimization import solve_ocp, visualize_ocp_result
from code.visuals.fatigue_trajectory import show_fatigue_trajectory


ROOT = Path(__file__).resolve().parent

# Demo settings. Change these values to configure the pipeline.
MODEL_NAME = "tsm_3DOF"
N_SAMPLES = 25
DATASET_TAG = "left_regular"
FORCE_MAGNITUDES = [20.0]
FORCE_DIRECTIONS = [[0.0, -1.0, 0.0]]
FORCE_INDEX = 0
HORIZON = 100
TIMESTEP = 0.01
OCP_TAG = "demo"  # Each run appends its fatigue model name.
FATIGUE_MODELS = ("3cc", "thresholded")  # "3cc" or "thresholded"
SHOW_GRADIENT_VISUALS = True
GRADIENT_RESOLUTION = 60
ACTIVATION_DYNAMICS = True
FILTER_SUCCESS = True
SHOW_TRAJECTORY_VISUAL = True
BLOCK_OCP_VISUAL = False
OCP_MUSCLES = [
    "DeltoideusScapula_M",
    "Supraspinatus_A",
    #"Infraspinatus_S",
    #"TeresMinor",
    #"PectoralisMajorThorax_M",
    #"LatissimusDorsi_M",
]


def main():
    os.chdir(ROOT)

    dataset_path = Path(
        build_msk_dataset(
            MODEL_NAME,
            n_samples=N_SAMPLES,
            magnitudes=FORCE_MAGNITUDES,
            directions=FORCE_DIRECTIONS,
            tag=DATASET_TAG,
            regular=True,
        )
    )
    MODEL_CONFIG[MODEL_NAME]["dataset_path"] = str(dataset_path)

    ocp_dir = ROOT / "results" / MODEL_NAME / "ocp"
    runs = {}
    for fatigue_model in FATIGUE_MODELS:
        tag = f"{OCP_TAG}_{fatigue_model}"
        result_path = ocp_dir / "results" / f"{tag}.npz"
        print(f"Optimizing {fatigue_model}: {result_path}")
        result = solve_ocp(
            MODEL_NAME,
            force_index=FORCE_INDEX,
            horizon=HORIZON,
            timestep=TIMESTEP,
            tag=tag,
            muscles=OCP_MUSCLES,
            fatigue_model=fatigue_model,
        )
        runs[fatigue_model] = (result, result_path)

        if SHOW_TRAJECTORY_VISUAL:
            visualize_ocp_result(
                MODEL_NAME,
                ocp_result_path=result_path,
                block=BLOCK_OCP_VISUAL,
            )

    if SHOW_GRADIENT_VISUALS:
        import matplotlib.pyplot as plt
        import opensim as osim
        from code.msk_data import MSKdataset

        model = osim.Model(MODEL_CONFIG[MODEL_NAME]["model_path"])
        dataset = MSKdataset.from_npz(dataset_path)
        for fatigue_model, (result, _) in runs.items():
            for muscle in result.muscle_names:
                figure, _ = show_fatigue_trajectory(
                    model, dataset, result,
                    muscle=muscle,
                    fatigue_model=fatigue_model,
                    force_index=FORCE_INDEX,
                    resolution=GRADIENT_RESOLUTION,
                    show=False,
                )
                figure.canvas.manager.set_window_title(f"{fatigue_model} | {muscle}")
        # Display both models and all selected muscles together, with time sliders.
        plt.show(block=True)

    comparisons = {}
    for fatigue_model, (_, result_path) in runs.items():
        validation_path = ocp_dir / "validation" / result_path.name
        MODEL_CONFIG[MODEL_NAME]["ocp_result_path"] = str(result_path)
        MODEL_CONFIG[MODEL_NAME]["ocp_validation_path"] = str(validation_path)

        build_validation_dataset(
            MODEL_NAME,
            str(dataset_path),
            str(result_path),
            force_index=FORCE_INDEX,
            activation_dynamics=ACTIVATION_DYNAMICS,
            filter_succes=FILTER_SUCCESS,
        )
        comparisons[fatigue_model] = compare_muscle_state_trajectories(
            MODEL_NAME,
            force_index=FORCE_INDEX,
        )
    return comparisons



if __name__ == "__main__":
    main()
