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


class MPCTests(unittest.TestCase):
    def make_ocp(self, fatigue_model=None, cost_options=None, *, moving_task=True):
        q = ca.MX.sym("q", 1)
        kinematics = SimpleNamespace(
            endpoint_function=ca.Function("endpoint", [q], [ca.vertcat(q, 0, 0)]),
            inverse_kinematics=lambda point: (np.array([point[0]]), None),
        )
        joints = JointStates(("q",), np.zeros(1), np.array([[-2., 2.]]),
                             np.zeros(1), np.zeros(1))
        conf = OCPconfig(6, 0.05, np.ones(1), np.ones(1), 1., 1.)
        for name, value in (cost_options or {}).items():
            setattr(conf, name, value)
        conf.s_opts = dict(conf.s_opts, print_level=0)
        return MusculoskeletalOCP(
            kinematics, SecondOrderIntegrator(1), None, Fatigue() if fatigue_model is None else fatigue_model,
            SphericalRegion(np.zeros(3), 0.5, moving=moving_task), joints, conf,
        )

    def test_plant_constant_acceleration(self):
        plant = JointPlant([1.], [2.])
        np.testing.assert_allclose(plant.step([3.], 0.2), [1.46])
        np.testing.assert_allclose(plant.velocities, [2.6])

    def test_fixed_task_center_stays_fixed_when_motion_is_rewarded(self):
        for moving in (False, True):
            with self.subTest(moving=moving):
                ocp = self.make_ocp(moving_task=moving)
                ocp.opti.minimize(ocp.opti.f - ocp.task_params[0, -1])
                ocp.solve(diagnostics=False, validate=False)
                if moving:
                    self.assertGreater(ocp.task_parameters_solution[0, -1], 0.01)
                else:
                    np.testing.assert_allclose(ocp.task_parameters_solution, 0., atol=1e-7)
                    np.testing.assert_allclose(ocp.task_param_rates_solution, 0., atol=1e-7)

    def test_feedback_velocity_and_fatigue(self):
        controller = MusculoskeletalMPC(self.make_ocp(), np.array([0., 0.4, 0.2]))
        controller.previous_control = np.array([2.])
        plant = JointPlant([0.], [0.4])
        positions = plant.step([2.], controller.sample_time)
        controller.update_state(positions)
        np.testing.assert_allclose(controller.state[:2], [positions[0], plant.velocities[0]])
        expected = 1 - 0.8 * np.exp(-0.02 * controller.sample_time)
        self.assertAlmostEqual(controller.state[2], expected, places=12)

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
        acceleration = controller.step(np.zeros(1))
        np.testing.assert_allclose(controller.result.controls[:, :2],
                                   np.repeat(acceleration[:, None], 2, axis=1), atol=1e-7)
        plant = JointPlant([0.], [0.])
        controller.step(plant.step(acceleration, controller.sample_time))
        np.testing.assert_allclose(controller.state[1:2], plant.velocities, atol=1e-7)

    def test_closed_loop_carries_task_and_fatigue(self):
        ocp = self.make_ocp()
        controller = MusculoskeletalMPC(ocp, np.array([0., 0., 0.2]))
        plant = JointPlant([0.], [0.])
        positions = plant.positions.copy()
        for _ in range(3):
            expected_task = controller.task_parameters.copy()
            acceleration = controller.step(positions)
            np.testing.assert_allclose(ocp.task_parameters_solution[:, 0], expected_task, atol=1e-7)
            np.testing.assert_allclose(controller.result.states[:, 0], controller.state, atol=1e-7)
            positions = plant.step(acceleration, controller.sample_time)
        self.assertGreater(controller.state[-1], 0.2)

    def test_each_feedback_update_solves_a_new_prediction(self):
        ocp = self.make_ocp()
        controller = MusculoskeletalMPC(ocp)
        plant = JointPlant([0.], [0.])
        with patch.object(ocp, "solve", wraps=ocp.solve) as solve:
            acceleration = controller.step(plant.positions)
            first_prediction = controller.result
            controller.step(plant.step(acceleration, controller.sample_time))
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
                positions = plant.positions.copy()
                for _ in range(3):
                    acceleration = controller.step(positions)
                    np.testing.assert_allclose(controller.result.states[:, 0],
                                               controller.state, atol=1e-7)
                    positions = plant.step(acceleration, controller.sample_time)
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
            if calls == 3:
                signal.raise_signal(signal.SIGINT)
            return positions

        previous_handler = signal.getsignal(signal.SIGINT)
        with TemporaryDirectory() as directory, \
                patch.multiple(main_mpc, ROOT=Path(directory), DURATION=None,
                               FATIGUE_MODEL="thresholded", SHOW_SIMBODY=False,
                               SHOW_FATIGUE=False), \
                patch.object(main_mpc.os, "chdir"), \
                patch.object(main_mpc.osim, "Model"), \
                patch.object(main_mpc.MusculoskeletalOCP, "from_model",
                             return_value=self.make_ocp()) as build, \
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

    def make_two_muscle_model(self, model_type):
        demand = np.linspace(0., 3., 4)
        dataset = MSKdataset(
            SimpleNamespace(joint_coordinate_names=("q",)),
            SimpleNamespace(joint_coordinates=np.linspace(-2., 2., 4)[:, None],
                            forces=np.zeros((1, 3))),
            SimpleNamespace(sample_success=np.ones((4, 1), dtype=bool),
                            sample_muscle_forces=np.column_stack((demand, demand[::-1]))[:, None, :]),
        )
        muscles = MuscleInfo(("a", "b"), np.array([0, 1]), np.array([10., 10.]))
        options = {"transition_width": 0.01} if model_type is Thresholded else {}
        return model_type(dataset, muscles, **options)

    def test_overlap_prefers_offloading_one_muscle(self):
        for model_type in (Thresholded, ThreeCompartment):
            with self.subTest(model=model_type.__name__):
                model = self.make_two_muscle_model(model_type)
                ocp = self.make_ocp(model, {"w_overlap": 1., "muscle_pairs": (("a", "b"),)})
                mixed = np.zeros(2 + model.n_states)
                offloaded = mixed.copy()
                offloaded[0] = 1.5
                index = ocp.cost_names.index("overlap")
                mixed_cost = float(ocp.stage_cost(mixed, [0.])[index])
                offloaded_cost = float(ocp.stage_cost(offloaded, [0.])[index])
                self.assertGreater(mixed_cost, offloaded_cost)
                self.assertGreaterEqual(offloaded_cost, 0.)

    def test_recovery_and_recruitment_are_not_positive_fatigue_rates(self):
        for model_type in (Thresholded, ThreeCompartment):
            with self.subTest(model=model_type.__name__):
                model = self.make_two_muscle_model(model_type)
                ocp = self.make_ocp(model, {"w_fatigue_rate": 1.})
                if model_type is Thresholded:
                    # First muscle is saturated and the unloaded second muscle recovers.
                    state = np.array([2., 0., 1., 0.2])
                else:
                    # Both fatigued fractions recover while active fractions are recruited.
                    state = np.array([0., 0., 0.2, 0.2, 0., 0.])
                    derivative = np.asarray(model.derivative([0.], state[2:])).ravel()
                    self.assertTrue((derivative[2:] > 0).all())
                rate_cost = float(ocp.stage_cost(state, [0.])[ocp.cost_names.index("fatigue_rate")])
                self.assertAlmostEqual(rate_cost, 0., places=12)
                # A fresh loaded muscle does incur positive fatigue-rate cost.
                if model_type is Thresholded:
                    state[2] = 0.2
                else:
                    state[2:4] = 0.
                    state[4:] = 0.2
                self.assertGreater(float(ocp.stage_cost(state, [0.])[1]), 0.)

    def test_negative_interpolated_force_cannot_reward_overlap(self):
        model = self.make_two_muscle_model(ThreeCompartment)
        model.force_interpolant = lambda q: ca.DM([-1., 2.])
        ocp = self.make_ocp(model, {"w_overlap": 1., "muscle_pairs": (("a", "b"),)})
        self.assertEqual(float(ocp.stage_cost(np.zeros(6), [0.])[2]), 0.)

    def test_objective_components_match_solver_for_both_models(self):
        for model_type in (Thresholded, ThreeCompartment):
            with self.subTest(model=model_type.__name__):
                model = self.make_two_muscle_model(model_type)
                ocp = self.make_ocp(model, {
                    "w_fatigue": 1., "w_fatigue_rate": 0.1, "w_overlap": 1.,
                    "w_control": 1e-3, "muscle_pairs": (("a", "b"),),
                })
                result = ocp.solve(diagnostics=False, validate=False)
                terms = ocp.evaluate_objective(result.states, result.controls)
                self.assertAlmostEqual(sum(terms.values()), result.objective, places=10)
                self.assertTrue(all(value >= -1e-12 for value in terms.values()))
                running = ocp.evaluate_objective(result.states, result.controls, include_terminal=False)
                longer = ocp.evaluate_objective(result.states, result.controls,
                                               include_terminal=False, timestep=2 * ocp.conf.dt)
                np.testing.assert_allclose(list(longer.values()), 2 * np.array(list(running.values())))

    def test_fatigue_only_cost_preserves_original_objective(self):
        ocp = self.make_ocp(cost_options={"w_fatigue": 0.1, "w_control": 0.})
        result = ocp.solve(diagnostics=False, validate=False)
        expected = 0.1 * ocp.conf.dt * np.sum(result.states[2:]**2)
        self.assertAlmostEqual(result.objective, expected, places=12)

    def test_swapping_fatigue_changes_the_preferred_loading_direction(self):
        for model_type in (Thresholded, ThreeCompartment):
            with self.subTest(model=model_type.__name__):
                model = self.make_two_muscle_model(model_type)
                ocp = self.make_ocp(model, {
                    "w_fatigue": 1., "w_fatigue_rate": 0.1, "w_overlap": 1.,
                    "w_control": 1e-3, "muscle_pairs": (("a", "b"),),
                })
                first = np.concatenate(([0., 0.], [0.5, 0.1],
                                        [0.15, 0.15] if model_type is ThreeCompartment else []))
                second = first.copy()
                second[2:4] = first[2:4][::-1]
                result_a = ocp.solve(first, diagnostics=False, validate=False)
                result_b = ocp.solve(second, diagnostics=False, validate=False)
                self.assertLess(result_a.controls[0, 0], 0.)
                self.assertGreater(result_b.controls[0, 0], 0.)


if __name__ == "__main__":
    unittest.main()
