"""Run with .venvs/opensim/bin/python -m unittest discover -s src/human_pose/test."""

import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@unittest.skipUnless(importlib.util.find_spec('opensim'), 'Requires OpenSim environment')
class GeometricAlignmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        from human_pose.opensim_viewer import PoseModel

        cls.np = np
        model = (Path(__file__).resolve().parents[3] / 'musculoskeletal_models'
                 / 'tsm_3DOF' / 'tsm_3DOF.osim')
        cls.viewer = PoseModel(model, visualize=False)

    def test_recovers_model_postures_from_geometric_angles(self):
        np = self.np
        for q in ([0, .4, .3], [.6, 1.2, 1], [1.5, .8, 1.5], [-.5, 1.8, .2]):
            with self.subTest(q=q):
                direction, bend = self.viewer.geometry(q)
                angles = [np.arctan2(direction[0], direction[2]),
                          np.arccos(-direction[1]), bend]
                self.assertLess(self.viewer.fit(angles), 1e-4)
                recovered, recovered_bend = self.viewer.geometry(self.viewer.q)
                np.testing.assert_allclose(recovered, direction, atol=1e-4)
                self.assertAlmostEqual(recovered_bend, bend, places=4)

    def test_unreachable_pose_respects_coordinate_limits(self):
        self.assertGreater(self.viewer.fit([0, 3.1, 3]), .15)
        self.assertTrue((self.viewer.q >= self.viewer.lower).all())
        self.assertTrue((self.viewer.q <= self.viewer.upper).all())

    def test_rejects_invalid_angles(self):
        for angles in ([0, 0], [0, float('nan'), 0], [0, 0, float('inf')]):
            with self.subTest(angles=angles), self.assertRaises(ValueError):
                self.viewer.fit(angles)


if __name__ == '__main__':
    unittest.main()
