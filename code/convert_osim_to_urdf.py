"""Convert an OpenSim model with the sibling ospi2urdf project.

Example:
  .venvs/ros/bin/python code/convert_osim_to_urdf.py \
    musculoskeletal_models/tsm_hand_4DOF/tsm_hand_4DOF.osim
"""

import argparse
import re
import shutil
import sys
import tempfile
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]


def _pinocchio_compat():
    """Provide the small Pinocchio API used by ospi2urdf's URDF writer.

    The writer only uses joint type names, velocity dimensions, and rotations.
    This avoids a binary Pinocchio dependency in the ROS Python environment.
    """
    module = types.ModuleType("pinocchio")

    class JointModelComposite:
        def __init__(self):
            self.joints = []

        @property
        def nv(self):
            return sum(joint.nv for joint in self.joints)

        def addJoint(self, joint):
            self.joints.append(joint)

    class JointModelFreeFlyer:
        nv = 6

    class JointModelSpherical:
        nv = 3

    class JointModelRX:
        nv = 1

    class JointModelRY:
        nv = 1

    class JointModelRZ:
        nv = 1

    class JointModelRevoluteUnaligned:
        nv = 1

        def __init__(self, x, y, z):
            self.axis = (x, y, z)

    class SE3:
        def __init__(self):
            self.rotation = np.eye(3)
            self._translation = np.zeros(3)

        @property
        def translation(self):
            return self._translation

        @translation.setter
        def translation(self, value):
            self._translation = np.asarray(value, dtype=float).reshape(3)

        @staticmethod
        def Identity():
            return SE3()

    module.JointModelComposite = JointModelComposite
    module.JointModelFreeFlyer = JointModelFreeFlyer
    module.JointModelSpherical = JointModelSpherical
    module.JointModelRX = JointModelRX
    module.JointModelRY = JointModelRY
    module.JointModelRZ = JointModelRZ
    module.JointModelRevoluteUnaligned = JointModelRevoluteUnaligned
    module.SE3 = SE3
    module.utils = types.SimpleNamespace(
        rotate=lambda axis, angle: Rotation.from_euler(axis, angle).as_matrix(),
        rpyToMatrix=lambda rpy: Rotation.from_euler(
            "xyz", np.asarray(rpy).reshape(3)
        ).as_matrix(),
    )
    return module


def _prepare_meshes(geometry, stl_dir):
    """Convert source meshes without invoking ospi2urdf's optional shell tools."""
    import vtk

    stl_dir.mkdir(parents=True)
    transform = np.dot(
        Rotation.from_euler("z", np.pi / 2).as_matrix(),
        Rotation.from_euler("x", np.pi / 2).as_matrix(),
    )
    corrections = {}
    readers = {
        ".vtp": vtk.vtkXMLPolyDataReader,
        ".obj": vtk.vtkOBJReader,
        ".stl": vtk.vtkSTLReader,
    }
    for source in sorted(geometry.iterdir()):
        if source.suffix.lower() not in readers:
            continue
        reader = readers[source.suffix.lower()]()
        reader.SetFileName(str(source))
        reader.Update()
        polydata = reader.GetOutput()
        if polydata.GetNumberOfPoints() == 0:
            raise ValueError("Mesh contains no points: {}".format(source))
        target = stl_dir / (source.stem + ".stl")
        if source.suffix.lower() == ".stl":
            shutil.copy2(source, target)
        else:
            triangles = vtk.vtkTriangleFilter()
            triangles.SetInputConnection(reader.GetOutputPort())
            writer = vtk.vtkSTLWriter()
            writer.SetFileName(str(target))
            writer.SetInputConnection(triangles.GetOutputPort())
            if writer.Write() != 1:
                raise RuntimeError("Could not write {}".format(target))
        x0, x1, y0, y1, z0, z1 = polydata.GetBounds()
        center = transform @ np.array([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
        corrections[source.stem] = {True: [], False: center.tolist()}
    if not corrections:
        raise ValueError("No VTP, OBJ, or STL meshes found in {}".format(geometry))
    return corrections


def _correct_weld_transforms(osim_tree, urdf):
    """Use both OpenSim offset frames for each fixed URDF body transform."""
    basis = np.dot(
        Rotation.from_euler("z", np.pi / 2).as_matrix(),
        Rotation.from_euler("x", np.pi / 2).as_matrix(),
    )
    contents = urdf.read_text()
    for joint in osim_tree.findall(".//JointSet/objects/WeldJoint"):
        frames = {
            frame.attrib["name"]: frame
            for frame in joint.findall("frames/PhysicalOffsetFrame")
        }
        parent_frame = frames[joint.findtext("socket_parent_frame").split("/")[-1]]
        child_frame = frames[joint.findtext("socket_child_frame").split("/")[-1]]

        def frame_transform(frame):
            translation = np.fromstring(frame.findtext("translation", "0 0 0"), sep=" ")
            angles = np.fromstring(frame.findtext("orientation", "0 0 0"), sep=" ")
            transform = np.eye(4)
            # OpenSim's frame-fixed XYZ sequence is represented by intrinsic XYZ.
            transform[:3, :3] = Rotation.from_euler("XYZ", angles).as_matrix()
            transform[:3, 3] = translation
            return transform

        body_transform = frame_transform(parent_frame) @ np.linalg.inv(
            frame_transform(child_frame))
        translation = basis @ body_transform[:3, 3]
        rotation = basis @ body_transform[:3, :3] @ basis.T
        rpy = Rotation.from_matrix(rotation).as_euler("xyz")
        xyz_text = " ".join("{:.12f}".format(value) for value in translation)
        rpy_text = " ".join("{:.12f}".format(value) for value in rpy)
        name = re.escape(joint.attrib["name"])
        pattern = r'(<joint name="{}" type="fixed">\s*<origin )xyz="[^"]+" rpy="[^"]+"'.format(name)
        contents, count = re.subn(
            pattern, lambda match: '{}xyz="{}" rpy="{}"'.format(
                match.group(1), xyz_text, rpy_text), contents)
        if count != 1:
            raise ValueError("Expected one fixed URDF joint named {}".format(joint.attrib["name"]))
    urdf.write_text(contents)


def _disable_gazebo_gravity(urdf):
    """Keep generated human links free of gravity in Gazebo."""
    links = [link.attrib["name"] for link in ET.parse(str(urdf)).findall(".//link")]
    # Gazebo 9's URDF parser maps this tag to SDF link <gravity>false</gravity>.
    tags = "\n".join(
        '    <gazebo reference="{}"><turnGravityOff>true</turnGravityOff></gazebo>'.format(name)
        for name in links
    )
    contents, count = re.subn(r"</robot>", tags + "\n</robot>", urdf.read_text(), count=1)
    if count != 1:
        raise ValueError("Expected one closing robot tag in {}".format(urdf))
    urdf.write_text(contents)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="OpenSim .osim model")
    parser.add_argument(
        "--package", type=Path, default=ROOT / "ros_ws/src/human_musculoskeletal",
        help="ROS package receiving description/<model>/",
    )
    parser.add_argument(
        "--geometry", type=Path,
        help="Directory of VTP, OBJ, or STL meshes; defaults to Geometry beside the model",
    )
    parser.add_argument(
        "--converter", type=Path, default=ROOT.parent / "ospi2urdf",
        help="Path to the ospi2urdf checkout",
    )
    args = parser.parse_args()

    model = args.model.resolve()
    package = args.package.resolve()
    geometry = (args.geometry or model.parent / "Geometry").resolve()
    converter = args.converter.resolve()
    for path in (model, geometry, converter / "src/osim2urdf_parser.py", package / "package.xml"):
        if not path.exists():
            parser.error("Missing required path: {}".format(path))

    try:
        import pinocchio  # noqa: F401
    except (ImportError, OSError):
        sys.modules["pinocchio"] = _pinocchio_compat()

    sys.path.insert(0, str(converter / "src"))
    from osim2urdf_parser import Osim2URDF

    # The upstream writer also creates a launch file with machine-specific paths.
    # Generate into a matching temporary package and keep only its description.
    with tempfile.TemporaryDirectory(prefix="ospi2urdf-") as temp:
        stage = Path(temp) / package.name
        (stage / "launch").mkdir(parents=True)
        model_stage = Path(temp) / "model"
        model_stage.mkdir()
        tree = ET.parse(str(model))
        # OpenSim omits identity orientations on some offset frames; ospi2urdf
        # expects an explicit orientation element on every such frame.
        for frame in tree.findall(".//BodySet/objects/Body/components/PhysicalOffsetFrame"):
            if frame.find("orientation") is None:
                ET.SubElement(frame, "orientation").text = "0 0 0"
        tree.write(str(model_stage / model.name), encoding="utf-8", xml_declaration=True)
        stl_dir = stage / "description" / model.stem / "stl"
        corrections = _prepare_meshes(geometry, stl_dir)
        result = Osim2URDF(
            str(model_stage), [str(stl_dir), "stl"], model.name, str(stage),
            exactly_visual=True, exactly_collision=False, correc_stl_dic=corrections,
            verbose=False, trans_plugin=False, xacro=False, add_rviz=False,
            roslaunch_name=model.stem + "_temporary",
        )
        source = stage / "description" / result.model_name
        target = package / "description" / result.model_name
        urdf = source / result.urdf_filename
        _correct_weld_transforms(tree, urdf)
        _disable_gazebo_gravity(urdf)
        meshes = {
            Path(mesh.attrib["filename"]).name
            for mesh in ET.parse(str(urdf)).findall(".//mesh")
        }
        missing = meshes - {mesh.name for mesh in (source / "stl").glob("*.stl")}
        if missing:
            raise RuntimeError("Missing generated meshes: {}".format(sorted(missing)))
        for mesh in (source / "stl").glob("*.stl"):
            if mesh.name not in meshes:
                mesh.unlink()
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)
        print("Generated {} ({} meshes)".format(target / result.urdf_filename, len(meshes)))


if __name__ == "__main__":
    main()
