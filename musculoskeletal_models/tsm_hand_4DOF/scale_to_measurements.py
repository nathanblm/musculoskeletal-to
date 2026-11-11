"""Create the arm/hand model fitted to the measured right-arm dimensions."""

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import opensim as osim
from scipy.optimize import root


FOLDER = Path(__file__).resolve().parent
SOURCE = FOLDER / 'tsm_4DOF_arm26_hand.osim'
OUTPUT = FOLDER / 'tsm_4DOF_arm26_hand_scaled.osim'
MEASUREMENTS_CM = {
    'shoulder_tip_to_outer_elbow': 35.0,
    'outer_elbow_to_wrist': 27.0,
    'wrist_to_index_mcp': 9.0,
    'wrist_to_pinky_mcp': 8.5,
    'index_to_pinky_mcp': 6.5,
}


def xyz(vector):
    return np.array([vector[i] for i in range(3)])


# These anatomical surface markers existed in the original TSM model before
# the marker set was reduced to the five tracking landmarks.
anatomy = ET.parse(FOLDER.parent / 'tsm_4DOF/tsm_4DOF.osim').getroot()
surface_markers = {}
for name in ('acc', 'EpL'):
    marker = anatomy.find(f'.//Marker[@name="{name}"]')
    surface_markers[name] = (
        marker.findtext('socket_parent_frame'),
        np.fromstring(marker.findtext('location'), sep=' '))


def scale_arm(factors):
    model = osim.Model(str(SOURCE))
    state = model.initSystem()
    masses = {model.getBodySet().get(i).getName(): model.getBodySet().get(i).getMass()
              for i in range(model.getBodySet().getSize())}
    scales = osim.ScaleSet()
    for name, factor in [('humerus', factors[0]), ('ulna', factors[1]),
                         ('radius', factors[1])]:
        scale = osim.Scale()
        scale.setSegmentName(name)
        scale.setScaleFactors(osim.Vec3(1., float(factor), 1.))
        scale.setApply(True)
        scales.cloneAndAppend(scale)
    # Retain original segment masses; no subject mass measurements were supplied.
    model.scale(state, scales, False)
    # OpenSim 4.4 scales body masses with volume even with this flag. Restore
    # the source model's original masses and normalize the scaled inertias.
    for i in range(model.getBodySet().getSize()):
        body = model.getBodySet().get(i)
        mass = masses[body.getName()]
        ratio = mass / body.getMass()
        inertia = body.getInertia()
        values = np.r_[xyz(inertia.getMoments()), xyz(inertia.getProducts())] * ratio
        body.setInertia(osim.Inertia(*values))
        body.setMass(mass)
    state = model.initSystem()
    model.realizePosition(state)
    points = {}
    for name, (parent, point) in surface_markers.items():
        frame = osim.PhysicalFrame.safeDownCast(model.getComponent(parent))
        point = point * [1., factors[0] if name == 'EpL' else 1., 1.]
        points[name] = xyz(frame.findStationLocationInGround(state, osim.Vec3(*point)))
    wrist = xyz(model.getMarkerSet().get('wrist').getLocationInGround(state))
    distances = np.array([np.linalg.norm(points['acc'] - points['EpL']),
                          np.linalg.norm(points['EpL'] - wrist)])
    return model, state, distances


arm_targets = np.array([MEASUREMENTS_CM['shoulder_tip_to_outer_elbow'],
                        MEASUREMENTS_CM['outer_elbow_to_wrist']]) / 100
fit = root(lambda factors: scale_arm(factors)[2] - arm_targets, [1.3, 1.0])
assert fit.success, fit.message
model, state, arm_distances = scale_arm(fit.x)
hand = model.getBodySet().get('hand')
markers = model.getMarkerSet()
index = xyz(markers.get('index_mcp').findLocationInFrame(state, hand))
pinky = xyz(markers.get('pinky_mcp').findLocationInFrame(state, hand))

# Fit the measured wrist/index/pinky triangle in its existing palm plane.
# Hand thickness is unchanged. This also accommodates different MCP levels.
across = (index - pinky) / np.linalg.norm(index - pinky)
forward = index - across * np.dot(index, across)
forward /= np.linalg.norm(forward)
normal = np.cross(index, pinky)
normal /= np.linalg.norm(normal)
r_index, r_pinky, width = np.array([MEASUREMENTS_CM['wrist_to_index_mcp'],
                                  MEASUREMENTS_CM['wrist_to_pinky_mcp'],
                                  MEASUREMENTS_CM['index_to_pinky_mcp']]) / 100
along = (r_index**2 - r_pinky**2 + width**2) / (2 * width)
height = np.sqrt(r_index**2 - along**2)
new_index = along * across + height * forward
new_pinky = (along - width) * across + height * forward
hand_transform = (np.column_stack((new_index, new_pinky, normal))
                  @ np.linalg.inv(np.column_stack((index, pinky, normal))))

# Bake the same transform into copies of all hand meshes, including their
# original offset-frame registration. The unscaled mesh files stay untouched.
frame = osim.PhysicalOffsetFrame.safeDownCast(
    model.getComponent('/bodyset/hand/arm26_hand_geometry_offset'))
transform = frame.findTransformInBaseFrame()
rotation = np.array([[transform.R().get(i, j) for j in range(3)] for i in range(3)])
translation = xyz(transform.p())
for i in range(frame.getPropertyByName('attached_geometry').size()):
    mesh = osim.Mesh.safeDownCast(frame.upd_attached_geometry(i))
    matrix = np.eye(4)
    matrix[:3, :3] = hand_transform @ rotation @ np.diag(xyz(mesh.get_scale_factors()))
    matrix[:3, 3] = hand_transform @ translation
    filename = 'scaled_' + Path(mesh.get_mesh_file()).name
    # Preserve the original plain ASCII VTP layout. Modern VTK writers add
    # nested InformationKey metadata that the bundled Simbody renderer cannot parse.
    tree = ET.parse(FOLDER / 'Geometry' / mesh.get_mesh_file())
    data = tree.find('.//Points/DataArray')
    vertices = np.fromstring(data.text, sep=' ').reshape(-1, 3)
    vertices = vertices @ matrix[:3, :3].T + matrix[:3, 3]
    data.text = '\n' + '\n'.join(
        ' '.join(format(value, '.9g') for value in point) for point in vertices) + '\n'
    tree.write(FOLDER / 'Geometry' / filename, encoding='utf-8', xml_declaration=True)
    mesh.set_mesh_file(filename)
    mesh.set_scale_factors(osim.Vec3(1))
frame.set_translation(osim.Vec3(0))
frame.set_orientation(osim.Vec3(0))
for name in ('wrist', 'index_mcp', 'pinky_mcp'):
    marker = markers.get(name)
    marker.set_location(osim.Vec3(*(hand_transform @ xyz(marker.get_location()))))
hand.setMassCenter(osim.Vec3(*(hand_transform @ xyz(hand.getMassCenter()))))
inertia = hand.getInertia()
moments, products = xyz(inertia.getMoments()), xyz(inertia.getProducts())
inertia_matrix = np.array([
    [moments[0], products[0], products[1]],
    [products[0], moments[1], products[2]],
    [products[1], products[2], moments[2]],
])
second_moment = np.trace(inertia_matrix) / 2 * np.eye(3) - inertia_matrix
second_moment = hand_transform @ second_moment @ hand_transform.T
inertia_matrix = np.trace(second_moment) * np.eye(3) - second_moment
hand.setInertia(osim.Inertia(inertia_matrix[0, 0], inertia_matrix[1, 1],
                            inertia_matrix[2, 2], inertia_matrix[0, 1],
                            inertia_matrix[0, 2], inertia_matrix[1, 2]))
model.setName('tsm_4DOF_arm26_hand_scaled')
model.finalizeFromProperties()
model.finalizeConnections()
model.printToXML(str(OUTPUT))

report = {
    'measurements_cm': MEASUREMENTS_CM,
    'surface_landmarks': {'shoulder_tip': 'acc (clavicle)', 'outer_elbow': 'EpL (humerus)'},
    'length_scale_factors': {'humerus': float(fit.x[0]), 'ulna_and_radius': float(fit.x[1])},
    'hand_transform_in_hand_frame': hand_transform.tolist(),
    'mass_policy': 'Original segment masses retained; inertias adjusted for geometric scaling.',
    'anatomical_accuracy': 'Approximate surface-landmark fit from tape measurements.',
}
(FOLDER / 'scaling_measurements.json').write_text(json.dumps(report, indent=2) + '\n')
print('Created', OUTPUT)
print('Upper arm / forearm scale factors:', fit.x)
print('Surface-landmark distances (cm):', arm_distances * 100)
