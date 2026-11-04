"""Small MPC checks independent of model files and generated dynamics."""

from types import SimpleNamespace
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import casadi as ca
import numpy as np

from code.msk_data import JointStates, MuscleInfo, MSKdataset
from code.msk_fatiguemodel import Thresholded, ThreeCompartment
from code.msk_mpc import MusculoskeletalMPC
from code.msk_optimization import MusculoskeletalOCP, OCPconfig, SecondOrderIntegrator
from code.msk_plant import JointPlant
from code.msk_rmr_feedback import RMRFatigueObserver
from code.tasks import SphericalRegion


class Fatigue:
    n_states = 1
    muscles = SimpleNamespace(names=("muscle",))
    n_muscles = 1

    def initial_state(self):
        return np.zeros(1)

    def derivative(self, q, state):
        return 0.02 * (1 - state)

    def fatigue(self, state):
        return state

    def rates_from_forces(self, forces):
        return np.full_like(forces, 0.02)

    def _numerical_derivative(self, state, rates, forces):
        return rates * (1 - state)


class ForceSolver:
    muscleNames = ("muscle",)

    def __init__(self):
        self.calls = []

    def solve(self, **kwargs):
        self.calls.append(kwargs)
        return None, None, np.array([2.]), {"success": True}


class MPCTests(unittest.TestCase):
    def observer(self, controller):
        return RMRFatigueObserver(ForceSolver(), controller.ocp.fatigue_model,
                                  controller.state[controller.n_joint:], np.zeros(3))

    def make_ocp(self, fatigue_model=None):
        q = ca.MX.sym("q", 1)
        kinematics = SimpleNamespace(
            endpoint_function=ca.Function("endpoint", [q], [ca.vertcat(q, 0, 0)]),
            inverse_kinematics=lambda point: (np.array([point[0]]), None),
        )
        joints = JointStates(("q",), np.zeros(1), np.array([[-2., 2.]]),
                             np.zeros(1), np.zeros(1))
        conf = OCPconfig(6, 0.05, np.ones(1), np.ones(1), 1., 1.)
        conf.s_opts = dict(conf.s_opts, print_level=0)
        return MusculoskeletalOCP(
            kinematics, SecondOrderIntegrator(1), None, Fatigue() if fatigue_model is None else fatigue_model,
            SphericalRegion(np.zeros(3), 0.5), joints, conf,
        )

    def test_plant_constant_acceleration(self):
        plant = JointPlant([1.], [2.])
        positions, velocities = plant.step([3.], 0.2)
        np.testing.assert_allclose(positions, [1.46])
        np.testing.assert_allclose(velocities, [2.6])

    def test_feedback_velocity_and_fatigue(self):
        controller = MusculoskeletalMPC(self.make_ocp(), np.array([0., 0.4, 0.2]))
        controller.update_state([0.], [0.7], [0.2])
        np.testing.assert_allclose(controller.state, [0., 0.7, 0.2])
        observer = self.observer(controller)
        observer.advance([0.], [0.4], [2.], controller.sample_time)
        controller.previous_control = np.array([2.])
        plant = JointPlant([0.], [0.4])
        positions, velocities = plant.step([2.], controller.sample_time)
        # Feedback may differ from the velocity predicted by the last control.
        velocities += 0.3
        controller.update_state(positions, velocities, observer.state)
        np.testing.assert_allclose(controller.state[:2], [positions[0], velocities[0]])
        expected = 1 - 0.8 * np.exp(-0.02 * controller.sample_time)
        self.assertAlmostEqual(controller.state[2], expected, places=12)

    def test_rmr_feedback_preserves_history_and_bypasses_maps(self):
        controller = MusculoskeletalMPC(self.make_ocp(), np.array([0., 0., 0.2]))
        observer = self.observer(controller)
        with patch.object(controller.ocp.fatigue_model, "derivative",
                          side_effect=AssertionError("Feedback must not use the map")):
            for i in range(5):
                observer.advance([i * 0.01], [0.1], [0.2], 0.01)
        np.testing.assert_allclose([call["time"] for call in observer.solver.calls],
                                   np.arange(5) * 0.01)
        np.testing.assert_allclose(observer.state, [1 - 0.8 * np.exp(-0.02 * 0.05)])
        np.testing.assert_allclose(observer.solver.calls[-1]["position"], [0.04])
        np.testing.assert_allclose(observer.solver.calls[-1]["acceleration"], [0.2])

    def test_rmr_feedback_selects_muscles_by_name(self):
        controller = MusculoskeletalMPC(self.make_ocp())
        solver = ForceSolver()
        solver.muscleNames = ("other", "muscle")
        solver.solve = lambda **kwargs: (None, None, np.array([99., 3.]), {"success": True})
        observer = RMRFatigueObserver(solver, controller.ocp.fatigue_model, [0.], [1., 2., 3.])
        with patch.object(observer.fatigue_model, "rates_from_forces", wraps=observer.fatigue_model.rates_from_forces) as rates:
            observer.advance([0.], [0.], [0.], 0.01)
        np.testing.assert_allclose(rates.call_args.args[0], [3.])
        np.testing.assert_allclose(observer.wrench, [1., 2., 3., 0., 0., 0.])

    def test_rmr_failure_does_not_advance_fatigue(self):
        observer = self.observer(MusculoskeletalMPC(self.make_ocp()))
        observer.solver.solve = lambda **kwargs: (None, None, None, {"success": False, "message": "failed"})
        with self.assertRaisesRegex(RuntimeError, "RMR feedback failed"):
            observer.advance([0.], [0.], [0.], 0.01)
        np.testing.assert_array_equal(observer.state, [0.])
        self.assertEqual(observer.time, 0.)

    def test_repeated_solve_fixes_state_without_adding_constraints(self):
        ocp = self.make_ocp()
        count = ocp.opti.g.numel()
        initial = np.array([0.1, 0.2, 0.3])
        for position in (0.1, 0.15):
            initial[0] = position
            result = ocp.solve(initial, diagnostics=False, validate=False)
            np.testing.assert_allclose(result.states[:, 0], initial, atol=1e-7)
            self.assertEqual(ocp.opti.g.numel(), count)

    def test_longer_update_holds_acceleration(self):
        controller = MusculoskeletalMPC(self.make_ocp(), steps_per_update=2)
        acceleration = controller.step(np.zeros(1), np.zeros(1), controller.state[controller.n_joint:])
        np.testing.assert_allclose(controller.result.controls[:, :2],
                                   np.repeat(acceleration[:, None], 2, axis=1), atol=1e-7)
        plant = JointPlant([0.], [0.])
        controller.step(*plant.step(acceleration, controller.sample_time),
                            controller.state[controller.n_joint:])
        np.testing.assert_allclose(controller.state[1:2], plant.velocities, atol=1e-7)

    def test_closed_loop_carries_task_and_fatigue(self):
        ocp = self.make_ocp()
        controller = MusculoskeletalMPC(ocp, np.array([0., 0., 0.2]))
        plant = JointPlant([0.], [0.])
        observer = self.observer(controller)
        positions, velocities = plant.positions.copy(), plant.velocities.copy()
        for _ in range(3):
            expected_task = controller.task_parameters.copy()
            acceleration = controller.step(positions, velocities, observer.state)
            np.testing.assert_allclose(ocp.task_parameters_solution[:, 0], expected_task, atol=1e-7)
            np.testing.assert_allclose(controller.result.states[:, 0], controller.state, atol=1e-7)
            observer.advance(positions, velocities, acceleration, controller.sample_time)
            positions, velocities = plant.step(acceleration, controller.sample_time)
        self.assertGreater(controller.state[-1], 0.2)

    def test_each_feedback_update_solves_a_new_prediction(self):
        ocp = self.make_ocp()
        controller = MusculoskeletalMPC(ocp)
        plant = JointPlant([0.], [0.])
        with patch.object(ocp, "solve", wraps=ocp.solve) as solve:
            acceleration = controller.step(plant.positions, plant.velocities, controller.state[controller.n_joint:])
            first_prediction = controller.result
            controller.step(*plant.step(acceleration, controller.sample_time),
                            controller.state[controller.n_joint:])
        self.assertEqual(solve.call_count, 2)
        self.assertIsNot(controller.result, first_prediction)
        self.assertTrue(np.isfinite(controller.prediction_revision))

    def test_both_fatigue_models_work_in_closed_loop(self):
        dataset = MSKdataset(
            SimpleNamespace(joint_coordinate_names=("q",)),
            SimpleNamespace(joint_coordinates=np.linspace(-2., 2., 4)[:, None], forces=np.zeros((1, 3))),
            SimpleNamespace(sample_success=np.ones((4, 1), dtype=bool),
                            sample_muscle_forces=np.full((4, 1, 1), 2.)),
        )
        muscles = MuscleInfo(("muscle",), np.array([0]), np.array([10.]))
        for model_type, options in ((Thresholded, {}),
                                    (Thresholded, {"transition_width": 0.01}),
                                    (ThreeCompartment, {})):
            with self.subTest(model=model_type.__name__, options=options):
                fatigue = model_type(dataset, muscles, **options)
                controller = MusculoskeletalMPC(self.make_ocp(fatigue))
                plant = JointPlant([0.], [0.])
                observer = self.observer(controller)
                positions, velocities = plant.positions.copy(), plant.velocities.copy()
                for _ in range(3):
                    acceleration = controller.step(positions, velocities, observer.state)
                    np.testing.assert_allclose(controller.result.states[:, 0],
                                               controller.state, atol=1e-7)
                    observer.advance(positions, velocities, acceleration, controller.sample_time)
                    positions, velocities = plant.step(acceleration, controller.sample_time)
                self.assertEqual(controller.state.size, 2 + fatigue.n_states)
                self.assertTrue(np.isfinite(controller.state).all())
                self.assertGreater(controller.state[2], 0.)

    def test_threshold_transition_matches_numerical_model(self):
        forces = np.linspace(0.5, 1.5, 4).reshape(4, 1, 1)
        dataset = MSKdataset(
            SimpleNamespace(joint_coordinate_names=("q",)),
            SimpleNamespace(joint_coordinates=np.linspace(-2., 2., 4)[:, None],
                            forces=np.zeros((1, 3))),
            SimpleNamespace(sample_success=np.ones((4, 1), dtype=bool),
                            sample_muscle_forces=forces),
        )
        muscles = MuscleInfo(("muscle",), np.array([0]), np.array([10.]))
        fatigue = Thresholded(dataset, muscles, transition_width=0.01)
        q = ca.MX.sym("posture")
        derivative = fatigue.derivative(q, ca.DM([0.2]))
        gradient = ca.Function("gradient", [q], [ca.jacobian(derivative, q)])
        for position in (-0.1, 0., 0.1):
            rate = np.asarray(fatigue.rate_interpolant(position)).ravel()
            force = np.asarray(fatigue.force_interpolant(position)).ravel()
            np.testing.assert_allclose(
                np.asarray(fatigue.derivative(position, np.array([0.2]))).ravel(),
                fatigue._numerical_derivative(np.array([0.2]), rate, force),
            )
            self.assertTrue(np.isfinite(float(gradient(position))))

    def test_indefinite_run_stops_and_saves_on_interrupt(self):
        import main_mpc

        original_step = JointPlant.step
        calls = 0

        def step_and_stop(plant, acceleration, dt):
            nonlocal calls
            positions = original_step(plant, acceleration, dt)
            calls += 1
            if calls == 15:
                signal.raise_signal(signal.SIGINT)
            return positions

        previous_handler = signal.getsignal(signal.SIGINT)
        with TemporaryDirectory() as directory, \
                patch.multiple(main_mpc, ROOT=Path(directory), DURATION=None,
                               FATIGUE_MODEL="thresholded", RMR_TIMESTEP=0.01, SHOW_SIMBODY=False,
                               SHOW_FATIGUE=False), \
                patch.object(main_mpc.os, "chdir"), \
                patch.object(main_mpc.osim, "Model"), \
                patch.object(main_mpc.MusculoskeletalOCP, "from_model",
                             return_value=self.make_ocp()) as build, \
                patch.object(main_mpc.RMRFatigueObserver, "from_model",
                             side_effect=lambda model, fatigue, state: RMRFatigueObserver(
                                 ForceSolver(), fatigue, state, np.zeros(3))), \
                patch.object(JointPlant, "step", step_and_stop):
            result = main_mpc.main()
            self.assertEqual(result.controls.shape[1], 3)
            self.assertEqual(result.states.shape[1], 4)
            self.assertEqual(result.fatigue_model, "thresholded")
            self.assertIs(build.call_args.kwargs["fatigue_model_type"], Thresholded)
            output = Path(directory) / "results" / main_mpc.MODEL_NAME / "mpc" / "results"
            self.assertTrue((output / f"{result.tag}.npz").exists())
            self.assertTrue((output / f"{result.tag}_timing.npz").exists())
        self.assertEqual(signal.getsignal(signal.SIGINT), previous_handler)


if __name__ == "__main__":
    unittest.main()
