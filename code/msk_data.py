from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

MODEL_CONFIG = {
    "arm26_3DOF": {
        "model_path": "musculoskeletal_models/arm26_3DOF/arm26_3DOF.osim",
        "dataset_path": "results/arm26_3DOF/datasets/msk_dataset_p1e5_m1_d1_fx-20_fy0_fz0.npz",
        "f_dynamics_func_path": "musculoskeletal_models/arm26_3DOF/fd_dynamics/F.casadi",
        "i_dynamics_func_path": "musculoskeletal_models/arm26_3DOF/inv_dynamics/F.so",
        "ocp_result_path": "results/arm26_3DOF/ocp/results/test.npz",
        "ocp_validation_path": "results/arm26_3DOF/ocp/validation/test.npz",
        "reference_marker": "r_acromion",
        "agonists": ["TRImed", "TRIlat", "TRIlong"],
        "antagonists": ["BICshort", "BIClong", "BRA"],
    },
    "arm26_4DOF": {
        "model_path": "musculoskeletal_models/arm26_4DOF/arm26_4DOF.osim",
        "dataset_path": "results/arm26_4DOF/datasets/msk_dataset_p1e5_m1_d1_fx-20_fy0_fz0.npz",
        "f_dynamics_func_path": "musculoskeletal_models/arm26_4DOF/fd_dynamics/F.casadi",
        "i_dynamics_func_path": "musculoskeletal_models/arm26_4DOF/inv_dynamics/F.so",
        "ocp_result_path": "results/arm26_4DOF/ocp/results/test.npz",
        "ocp_validation_path": "results/arm26_4DOF/ocp/validation/test.npz",
        "reference_marker": "r_acromion",
        "agonists": ["TRImed", "TRIlat", "TRIlong"],
        "antagonists": ["BICshort", "BIClong", "BRA"],
    },
    "tsm_4DOF": {
        "model_path": "musculoskeletal_models/tsm_4DOF/tsm_4DOF.osim",
        "dataset_path": "results/tsm_4DOF/datasets/msk_dataset_p1e5_m1_d1_left_regular.npz",
        "f_dynamics_func_path": "musculoskeletal_models/tsm_4DOF/fd_dynamics/F.casadi",
        "i_dynamics_func_path": "musculoskeletal_models/tsm_4DOF/inv_dynamics/F.so",
        "ocp_result_path": "results/tsm_4DOF/ocp/results/test.npz",
        "ocp_validation_path": "results/tsm_4DOF/ocp/validation/test.npz",
        "reference_marker": "ac",
        "agonists": ["TeresMinor"],
        "antagonists": ["Supraspinatus_A"], #Supraspinatus_P", 
    },
    "tsm_3DOF": {
        "model_path": "musculoskeletal_models/tsm_3DOF/tsm_3DOF.osim",
        "dataset_path": "results/tsm_3DOF/datasets/msk_dataset_p3e4_m1_d1_left_regular.npz",
        "f_dynamics_func_path": "musculoskeletal_models/tsm_3DOF/fd_dynamics/F.casadi",
        "i_dynamics_func_path": "musculoskeletal_models/tsm_3DOF/inv_dynamics/F.so",
        "ocp_result_path": "results/tsm_3DOF/ocp/results/test.npz",
        "ocp_validation_path": "results/tsm_3DOF/ocp/validation/test.npz",
        "interpolants_path": "results/tsm_3DOF/datasets/interpolants.npz",
        "reference_marker": "ac",
        "agonists": ["BIC_long", "BIC_brevis"],
        "antagonists": ["TRIlong"],
    },
}


def _cast_millard(muscle):
    import opensim as osim

    cast = osim.Millard2012EquilibriumMuscle.safeDownCast(muscle)
    if cast is None:
        raise TypeError(
            f"Expected Millard2012EquilibriumMuscle for "
            f"'{muscle.getName()}'; got {muscle.getConcreteClassName()}."
        )
    return cast

@dataclass
class ModelInfo:
    joint_coordinate_names: tuple[str, ...]     # len: n_coordinates
    joint_coordinate_limits: np.ndarray         # (n_coordinates, 2)
    muscle_names: tuple[str, ...]               # len: n_muscles
    muscle_max_forces: np.ndarray               # (n_muscles,)
    actuator_names: tuple[str, ...]             # len: n_actuators
    muscle_actuator_indices: np.ndarray         # (n_muscles,)
    prescribed_force_names: tuple[str, ...]     # len: n_prescribed_forces
    prescribed_force_indices: tuple[int, ...]   # len: n_prescribed_forces
    agonist_muscles: tuple[str, ...] = ()       # len: n_agonists
    antagonist_muscles: tuple[str, ...] = ()   # len: n_antagonists

    def __str__(self):
        return (
            f"ModelInfo(\n"
            f"  joint_coordinate_names={self.joint_coordinate_names},\n"
            f"  joint_coordinate_limits={self.joint_coordinate_limits},\n"
            f"  muscle_names={self.muscle_names},\n"
            f"  muscle_max_forces={self.muscle_max_forces},\n"
            f"  actuator_names={self.actuator_names},\n"
            f"  muscle_actuator_indices={self.muscle_actuator_indices},\n"
            f"  prescribed_force_names={self.prescribed_force_names},\n"
            f"  prescribed_force_indices={self.prescribed_force_indices},\n"
            f"  agonist_muscles={self.agonist_muscles},\n"
            f"  antagonist_muscles={self.antagonist_muscles}\n"
            f")"
        )

    @classmethod
    def from_model(cls, model, *, agonists=(), antagonists=()):
        coordinate_set = model.getCoordinateSet()
        joint_coordinate_names = tuple(
            coordinate.getName() 
            for coordinate in coordinate_set 
            if not coordinate.get_locked()
        )
        joint_coordinate_limits = np.asarray([
            [coordinate.getRangeMin(), coordinate.getRangeMax()]
            for coordinate in coordinate_set
            if not coordinate.get_locked()
        ])

        muscle_set = model.getMuscles()
        muscle_names = tuple(
            _cast_millard(muscle).getName()
            for muscle in muscle_set
        )
        muscle_max_forces = np.asarray([
            _cast_millard(muscle).getMaxIsometricForce()
            for muscle in muscle_set
        ])
        agonist_muscles = tuple(agonists)
        antagonist_muscles = tuple(antagonists)
        unknown = (set(agonist_muscles) | set(antagonist_muscles)) - set(muscle_names)
        if unknown:
            raise ValueError(f"Unknown muscles in agonist/antagonist groups: {sorted(unknown)}")
        if set(agonist_muscles).intersection(antagonist_muscles):
            raise ValueError("A muscle cannot belong to both agonist and antagonist groups.")

        actuator_set = model.getActuators()
        actuator_names = tuple(
            actuator.getName()
            for actuator in actuator_set
        )
        muscle_actuator_indices = np.asarray([
            actuator_set.getIndex(actuator_name)
            for actuator_name in muscle_names
        ])

        force_set = model.getForceSet()
        prescribed_force_names = tuple(
            force.getName()
            for force in force_set
            if force.getConcreteClassName() == "PrescribedForce"
        )
        prescribed_force_indices = tuple([
            force_set.getIndex(force)
            for force in force_set
            if force.getConcreteClassName() == "PrescribedForce"
        ])
        
        return cls(
            joint_coordinate_names=joint_coordinate_names,
            joint_coordinate_limits=joint_coordinate_limits,
            muscle_names=muscle_names,
            muscle_max_forces=muscle_max_forces,
            actuator_names=actuator_names,
            muscle_actuator_indices=muscle_actuator_indices,
            prescribed_force_names=prescribed_force_names,
            prescribed_force_indices=prescribed_force_indices,
            agonist_muscles=agonist_muscles,
            antagonist_muscles=antagonist_muscles,
        )

@dataclass
class SampleInputs:
    joint_coordinates: np.ndarray               # (n_position_samples, n_coordinates)
    cartesian_coordinates: np.ndarray           # (n_position_samples, 3)
    force_magnitudes: np.ndarray                # (n_magnitudes,)
    force_directions: np.ndarray                # (n_directions, 3)
    forces: np.ndarray                          # (n_magnitudes * n_directions, 3)

    @classmethod
    def empty(cls, n_joint_coordinates):
        return cls(
            joint_coordinates=np.empty((0, n_joint_coordinates)),
            cartesian_coordinates=np.empty((0, 3)),
            force_magnitudes=np.empty((0,)),
            force_directions=np.empty((0, 3)),
            forces=np.empty((0, 3))
        )

@dataclass
class SampleResults:
    sample_activations: np.ndarray                  # (n_samples, n_forces, n_muscles)
    sample_muscle_forces: np.ndarray                # (n_samples, n_forces, n_muscles)
    sample_success: np.ndarray                      # (n_samples, n_forces)
    sample_fatigue_rates: np.ndarray                # (n_samples, n_forces, n_muscles)
    sample_fatigue_dominance: np.ndarray            # (n_samples, n_forces)

    @classmethod
    def empty(cls):
        return cls(
            sample_activations=np.empty((0, 0, 0)),
            sample_muscle_forces=np.empty((0, 0, 0)),
            sample_success=np.empty((0, 0), dtype=bool),
            sample_fatigue_rates=np.empty((0, 0, 0)),
            sample_fatigue_dominance=np.empty((0, 0))
        )

@dataclass
class FatigueParameters:
    capacity: np.ndarray                            # C, (n_muscles,)
    recovery: float                                 # R
    threshold: np.ndarray                           # (n_muscles,)

    @classmethod
    def from_max_forces(cls, max_forces, *, T=10.0, th_frac=0.01, R=300.0):
        # based on formula from https://doi.org/10.1016/j.rcim.2019.01.013
        max_forces = np.asarray(max_forces, dtype=float)
        capacity = - (T * max_forces) / np.log(1 - 0.993)
        threshold = th_frac * max_forces
        params = cls(
            capacity=capacity,
            recovery=R,
            threshold=threshold,
        )
        return params

@dataclass
class RMRconfig:
    muscle_weights: np.ndarray
    reserve_actuator_weights: np.ndarray
    weights: np.ndarray

    @classmethod
    def from_model_info(cls, model_info: ModelInfo, *, reserve_actuator_weight=1.0, muscle_weight=1.0):
        n_muscles = len(model_info.muscle_names)
        n_reserve_actuators = len(model_info.actuator_names) - n_muscles

        muscle_weights = np.full(n_muscles, muscle_weight)
        reserve_actuator_weights = np.full(n_reserve_actuators, reserve_actuator_weight)
        weights = np.concatenate((muscle_weights, reserve_actuator_weights))

        return cls(
            muscle_weights=muscle_weights,
            reserve_actuator_weights=reserve_actuator_weights,
            weights=weights
        )

@dataclass
class FatigueValidationData:
    tag: str
    force_index: int        # currently only supports nearest-neighbor lookup for joint coordinates.

    time: np.ndarray
    joint_positions: np.ndarray
    endpoint_forces: np.ndarray
    

    muscle_forces: np.ndarray
    fatigue_rates: np.ndarray
    success: np.ndarray
    muscle_states: Optional[np.ndarray] = None

    @classmethod
    def empty(cls, n_muscles):
        return cls(
            tag="",
            force_index=0,
            time=np.empty((0,)),
            joint_positions=np.empty((0, 0)),
            endpoint_forces=np.empty((0, 3)),
            muscle_forces=np.empty((0, n_muscles)),
            fatigue_rates=np.empty((0, n_muscles)),
            success=np.empty((0,), dtype=bool),
            muscle_states=None
        )

    @classmethod
    def from_npz(cls, path):
        """Load a trusted archive containing a FatigueValidationData object."""
        with np.load(path, allow_pickle=True) as archive:
            return archive["data"].item()

    def save_npz(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, data=self)
        return path

@dataclass
class MSKdataset:
    info: ModelInfo
    samples: SampleInputs
    results: SampleResults
    fatigue_params: FatigueParameters

    @classmethod
    def from_npz(cls, path):
        """Load a trusted archive containing a pickled MSKdataset."""
        with np.load(path, allow_pickle=True) as archive:
            dataset = archive["dataset"].item()
        return dataset

    def save_npz(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, dataset=self)
        return path

@dataclass
class OCPResult:
    time: np.ndarray
    states: np.ndarray                      # (n_states, n_time_points)
    val_states: np.ndarray                  # (n_states, n_time_points)

    # for forward simulation of muscle states using nearest neighbor evaluation of discrete rate and force maps
    # tells us something about the accuracy of the continuous version the fatigue map wrt the discrete version

    val_states: np.ndarray                  # (n_states, n_time_points)
    pos_indices: np.ndarray                 # (n_coordinates,)
    controls: np.ndarray                    # (n_controls, n_time_points - 1)
    endpoint_positions: np.ndarray          # (n_endpoint_dims, n_time_points)
    reference: np.ndarray                   # (n_endpoint_dims, n_time_points)
    objective: float
    tag: str = ""
    muscle_names: tuple[str, ...] = ()

    @classmethod
    def from_npz(cls, path):
        """Load a trusted archive containing a pickled OCPResult."""
        with np.load(path, allow_pickle=True) as archive:
            result = archive["ocp_result"].item()
        return result

    def save_npz(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, ocp_result=self)
        return path

    @classmethod
    def empty(cls):
        return cls(
            time=np.empty((0,)),
            states=np.empty((0, 0)),
            val_states=np.empty((0, 0)),
            pos_indices=np.empty((0,), dtype=int),
            controls=np.empty((0, 0)),
            endpoint_positions=np.empty((0, 0)),
            reference=np.empty((0, 0)),
            objective=float("nan"),
        )

    @property
    def rms_tracking_error(self) -> float:
        error = self.endpoint_positions - self.reference
        return float(np.sqrt(np.mean(np.sum(error**2, axis=0))))

@dataclass
class JointStates:
    names: tuple[str, ...]                  # len(n_coordinates)
    angles: np.ndarray                      # (n_coordinates,)
    angle_limits: np.ndarray                # (n_coordinates, 2)
    velocities: np.ndarray                  # (n_coordinates,)
    torques: np.ndarray                     # (n_coordinates,)

    @classmethod
    def from_model(cls, model):
        info = ModelInfo.from_model(model)

        return cls(
            names=info.joint_coordinate_names,
            angles=np.zeros(len(info.joint_coordinate_names)),
            angle_limits=info.joint_coordinate_limits,
            velocities=np.zeros(len(info.joint_coordinate_names)),
            torques=np.zeros(len(info.joint_coordinate_names)),
        )

    @property
    def size(self) -> int:
        return self.angles.size + self.velocities.size

    @property
    def x(self) -> np.ndarray:
        return np.column_stack((self.angles, self.velocities)).ravel()

@dataclass
class MuscleStates:
    names: tuple[str, ...]                  # len(n_muscles)
    indexes: np.ndarray                     # (n_muscles,)
    fatigue: np.ndarray                     # (n_muscles,)
    max_forces: np.ndarray                  # (n_muscles,)

    @classmethod
    def from_model(cls, model, *, muscles: list[str] = None):
        info = ModelInfo.from_model(model)

        if muscles is None:
            return cls(
                names=info.muscle_names,
                indexes=np.arange(len(info.muscle_names)),
                fatigue=np.zeros(len(info.muscle_names)),
                max_forces=info.muscle_max_forces,
            )

        names = []
        fatigue = []
        max_forces = []
        indexes = []
        for muscle_name in muscles:
            if muscle_name not in info.muscle_names:
                raise ValueError(f"Muscle '{muscle_name}' not found in model.")
            index = info.muscle_names.index(muscle_name)
            names.append(info.muscle_names[index])
            fatigue.append(0.0)
            max_forces.append(info.muscle_max_forces[index])
            indexes.append(index)

        return cls(
            names=tuple(names),
            fatigue=np.array(fatigue),
            max_forces=np.array(max_forces),
            indexes=np.array(indexes),
        )

    @property
    def size(self) -> int:
        return len(self.names)

@dataclass
class Interpolants:
    """Self-contained linear maps in saved joint and muscle order.

    Values use CasADi ordering: muscles vary fastest, then joint coordinate 0.
    Each axis retains its model coordinate units (radians for rotational joints).
    """
    axes_list: list[list[float]]
    fatigue_rate_grid: list[float]
    muscle_force_grid: list[float]
    joint_coordinate_names: tuple[str, ...]
    muscle_names: tuple[str, ...]
    force: np.ndarray

    @classmethod
    def empty(cls):
        return cls([], [], [], (), (), np.zeros(3))

    def interpolant(self, field="fatigue"):
        """Build the same linear map used by the fatigue model and its plots."""
        import casadi as ca
        values = {"fatigue": self.fatigue_rate_grid,
                  "muscle_force": self.muscle_force_grid}[field]
        return ca.interpolant(f"{field}_interpolator", "linear", self.axes_list, values)

    @classmethod
    def from_npz(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            names = tuple(archive["joint_coordinate_names"].tolist())
            return cls(
                axes_list=[archive[f"axis_{i}"].tolist() for i in range(len(names))],
                fatigue_rate_grid=archive["fatigue_rate_grid"].tolist(),
                muscle_force_grid=archive["muscle_force_grid"].tolist(),
                joint_coordinate_names=names,
                muscle_names=tuple(archive["muscle_names"].tolist()),
                force=archive["force"].copy(),
            )

    def save_npz(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Separate axis arrays also support unequal grid lengths without pickle.
        np.savez_compressed(
            path, **{f"axis_{i}": axis for i, axis in enumerate(self.axes_list)},
            fatigue_rate_grid=self.fatigue_rate_grid,
            muscle_force_grid=self.muscle_force_grid,
            joint_coordinate_names=self.joint_coordinate_names,
            muscle_names=self.muscle_names, force=self.force,
        )
        return path
