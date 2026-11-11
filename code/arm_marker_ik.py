"""Minimal OpenSim marker IK for shoulder-relative camera measurements."""

import numpy as np
import opensim as osim

from arm_joint_state import CAMERA_TO_OPENSIM, OpenSimArmJointState


class OpenSimArmMarkerIK:
    """Fit the model's existing virtual markers, assuming a fixed shoulder."""

    def __init__(self, model_path, *, alignment=CAMERA_TO_OPENSIM, depth_sign=1.0):
        self.model = osim.Model(str(model_path))
        self.state = self.model.initSystem()
        self.alignment = np.asarray(alignment)
        self.depth_sign = depth_sign
        coordinates = self.model.getCoordinateSet()
        self.coordinates = [coordinates.get(i) for i in range(coordinates.getSize())
                            if not coordinates.get(i).get_locked()]
        self.coordinate_names = [c.getName() for c in self.coordinates]
        for coordinate in self.coordinates:
            coordinate.setClamped(self.state, True)

        self.marker_names = ('HumHead_Center', 'centelbow', 'endpoint')
        self.model.realizePosition(self.state)
        shoulder = self.model.getMarkerSet().get(self.marker_names[0])
        self.shoulder = np.array([
            shoulder.getLocationInGround(self.state)[i] for i in range(3)])

    def estimate(self, measurement):
        """Assemble this frame, using the previous fitted pose as the start."""
        vectors = np.array([measurement.elbow, measurement.wrist], dtype=float)
        vectors[:, 2] *= self.depth_sign
        targets = np.vstack((self.shoulder,
                             self.shoulder + vectors @ self.alignment.T))

        labels = osim.StdVectorString()
        row = osim.RowVectorVec3(3)
        for i, name in enumerate(self.marker_names):
            labels.append(name)
            row[i] = osim.Vec3(*targets[i])
        table = osim.TimeSeriesTableVec3()
        table.setColumnLabels(labels)
        table.appendRow(0.0, row)  # OpenSim marker references default to metres.
        reference = osim.MarkersReference(table, osim.SetMarkerWeights())
        solver = osim.InverseKinematicsSolver(
            self.model, reference, osim.SimTKArrayCoordinateReference())
        solver.setAccuracy(1e-8)
        solver.assemble(self.state)
        return np.array([c.getValue(self.state) for c in self.coordinates])

    def estimate_from_people(self, people):
        """Use the first detected person's shoulder-relative arm landmarks."""
        if not people:
            return None
        measurement = OpenSimArmJointState._measurement_from_person(people[0])
        return None if measurement is None else self.estimate(measurement)
