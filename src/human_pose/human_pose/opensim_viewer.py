"""OpenSim-only worker; runs under Python 3.9 without importing ROS or MMPose."""

import argparse
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time

import numpy as np
from scipy.optimize import least_squares


class PoseModel:
    """Fit geometric arm angles in the thorax frame to TSM coordinates."""

    coordinate_names = ('plane_elv', 'shoulder_elv', 'elbow_flexion')

    def __init__(self, model_path, visualize=True):
        import opensim as osim

        self.osim = osim
        self.visualize = visualize
        base = Path(sys.base_prefix)
        os.environ.setdefault('SIMBODY_HOME', str(base / 'simbody'))
        model_path = Path(model_path).resolve()
        osim.ModelVisualizer.addDirToGeometrySearchPaths(str(model_path.parent / 'Geometry'))
        self.model = osim.Model(str(model_path))
        self.model.setUseVisualizer(visualize)
        self.state = self.model.initSystem()
        self.coordinates = [self.model.getCoordinateSet().get(name)
                            for name in self.coordinate_names]
        self.lower = np.array([c.getRangeMin() for c in self.coordinates])
        self.upper = np.array([c.getRangeMax() for c in self.coordinates])
        self.q = np.clip([c.getValue(self.state) for c in self.coordinates],
                         self.lower, self.upper)
        self.shoulder = self.model.getJointSet().get('GlenoHumeral').getChildFrame()
        self.elbow = self.model.getJointSet().get('elbow').getChildFrame()
        self.wrist = self.model.getJointSet().get('rc').getParentFrame()
        self.thorax = self.model.getBodySet().get('thorax')
        self.model.realizePosition(self.state)
        # OpenSim thorax axes: +X anterior, +Y superior, +Z right.
        self.basis = np.column_stack([
            self.array(self.thorax.expressVectorInGround(self.state, osim.Vec3(*axis)))
            for axis in np.eye(3)
        ])
        if visualize:
            self.visualizer = self.model.getVisualizer().updSimbodyVisualizer()
            self.model.getVisualizer().show(self.state)

    @staticmethod
    def array(vector):
        return np.array([vector.get(i) for i in range(3)])

    def geometry(self, q):
        for coordinate, value in zip(self.coordinates, q):
            coordinate.setValue(self.state, float(value), False)
        self.model.realizePosition(self.state)
        shoulder, elbow, wrist = [self.array(frame.getPositionInGround(self.state))
                                  for frame in (self.shoulder, self.elbow, self.wrist)]
        arm = elbow - shoulder
        forearm = wrist - elbow
        arm /= np.linalg.norm(arm)
        forearm /= np.linalg.norm(forearm)
        bend = np.arccos(np.clip(np.dot(arm, forearm), -1, 1))
        return self.basis.T @ arm, bend

    def fit(self, angles):
        angles = np.asarray(angles, dtype=float)
        if angles.shape != (3,) or not np.isfinite(angles).all():
            raise ValueError('Expected three finite pose angles in radians')
        plane, elevation, bend = angles
        target = np.array([np.sin(elevation) * np.sin(plane), -np.cos(elevation),
                           np.sin(elevation) * np.cos(plane)])

        def residual(q):
            direction, actual_bend = self.geometry(q)
            return np.r_[direction - target, actual_bend - bend]

        options = dict(bounds=(self.lower, self.upper), max_nfev=40,
                       ftol=1e-6, xtol=1e-6, gtol=1e-6)
        result = least_squares(residual, self.q, **options)
        # Warm starts track motion; alternative branches recover large pose jumps.
        if np.linalg.norm(result.fun) > 0.05:
            for seed in ([0, elevation, bend], [np.pi / 2, elevation, bend]):
                candidate = least_squares(
                    residual, np.clip(seed, self.lower, self.upper), **options,
                )
                if np.linalg.norm(candidate.fun) < np.linalg.norm(result.fun):
                    result = candidate
        self.q = result.x
        self.geometry(self.q)
        return float(np.linalg.norm(result.fun))

    def show(self, elapsed):
        self.state.setTime(elapsed)
        self.model.realizePosition(self.state)
        if self.visualize:
            self.model.getVisualizer().show(self.state)


def read_latest(messages):
    for line in sys.stdin:
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if messages.full():
            try:
                messages.get_nowait()
            except queue.Empty:
                pass
        messages.put(value)
    messages.put(None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    viewer = PoseModel(args.model)
    messages = queue.Queue(maxsize=1)
    threading.Thread(target=read_latest, args=(messages,), daemon=True).start()
    start = time.monotonic()
    last_warning = -float('inf')
    try:
        while True:
            angles = messages.get()
            if angles is None:
                break
            error = viewer.fit(angles)
            elapsed = time.monotonic() - start
            if error > 0.15 and elapsed - last_warning > 5:
                print('Pose exceeds TSM joint limits or fixed-shoulder geometry; '
                      f'showing closest posture (fit residual {error:.3f}).',
                      file=sys.stderr, flush=True)
                last_warning = elapsed
            viewer.show(elapsed)
    finally:
        viewer.visualizer.shutdown()


if __name__ == '__main__':
    main()
