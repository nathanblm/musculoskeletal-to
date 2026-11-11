"""Check the viewer's ROS adapter against a real headless OpenSim model."""

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import opensim
import pytest
from sensor_msgs.msg import JointState

from fatigue_control_ros.model_viewer import ModelViewer, configure_simbody


@pytest.mark.parametrize('directory', ['bin', 'libexec/simbody'])
def test_find_visualizer_in_runtime(tmp_path, monkeypatch, directory):
    """Find the executable beside the bundled bindings in both layouts."""
    monkeypatch.delenv('SIMBODY_HOME', raising=False)
    monkeypatch.setattr('sys.prefix', str(tmp_path / 'venv'))
    runtime = tmp_path / 'runtime'
    bindings = runtime / 'lib/python3.10/site-packages/opensim/__init__.py'
    executable = runtime / 'simbody' / directory / 'simbody-visualizer'
    executable.parent.mkdir(parents=True)
    executable.touch()
    executable.chmod(0o755)
    configure_simbody(SimpleNamespace(__file__=str(bindings)))
    assert Path(os.environ['SIMBODY_HOME']) == runtime / 'simbody'


def test_preserve_explicit_simbody_home(monkeypatch):
    """Honor the user's chosen Simbody installation."""
    monkeypatch.setenv('SIMBODY_HOME', '/custom/simbody')
    configure_simbody(opensim)
    assert os.environ['SIMBODY_HOME'] == '/custom/simbody'


@pytest.fixture
def viewer():
    """Use native model coordinates without opening a desktop window."""
    model_path = (Path(__file__).resolve().parents[3]
                  / 'musculoskeletal_models/tsm_3DOF/tsm_3DOF.osim')
    model = opensim.Model(str(model_path))
    state = model.initSystem()
    coordinates = model.getCoordinateSet()
    display = Mock()
    node = SimpleNamespace(
        model=Mock(wraps=model), state=state,
        coordinates={coordinates.get(i).getName(): coordinates.get(i)
                     for i in range(coordinates.getSize())
                     if not coordinates.get(i).get_locked()},
        get_logger=Mock(return_value=Mock()),
    )
    node.model.getVisualizer.return_value = display
    return node, display


def test_named_positions_update_native_model(viewer):
    """Reordered and partial messages preserve omitted model coordinates."""
    node, display = viewer
    names = list(node.coordinates)
    omitted = node.coordinates[names[1]]
    initial = omitted.getValue(node.state)
    msg = JointState(name=[names[-1], 'unknown', names[0]],
                     position=[0.4, 1.0, 0.2])
    ModelViewer.joint_state_callback(node, msg)
    for name, expected in [(names[-1], 0.4), (names[0], 0.2)]:
        assert (node.coordinates[name].getValue(node.state)
                == pytest.approx(expected))
    assert omitted.getValue(node.state) == initial
    display.show.assert_called_once_with(node.state)


@pytest.mark.parametrize('kind', [
    'empty', 'mismatch', 'duplicate', 'nan', 'unknown',
])
def test_invalid_messages_do_not_update_model(viewer, kind):
    """Malformed messages must not partially alter or render the pose."""
    node, display = viewer
    name = next(iter(node.coordinates))
    names, positions = {
        'empty': ([], []),
        'mismatch': ([name], []),
        'duplicate': ([name, name], [0.2, 0.3]),
        'nan': ([name], [float('nan')]),
        'unknown': (['unknown'], [0.2]),
    }[kind]
    ModelViewer.joint_state_callback(
        node, JointState(name=names, position=positions))
    node.model.assemble.assert_not_called()
    display.show.assert_not_called()
    node.get_logger.return_value.warning.assert_called_once()
