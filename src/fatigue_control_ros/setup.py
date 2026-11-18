from catkin_pkg.python_setup import generate_distutils_setup
from setuptools import setup

setup(**generate_distutils_setup(
    packages=[
        'fatigue_control_ros', 'code', 'code.visuals', 'code.rtmpose3d',
        'rmr_solver',
    ],
    package_dir={
        'fatigue_control_ros': 'fatigue_control_ros',
        'code': '../../code',
        'rmr_solver': '../../rmr_solver',
    },
    package_data={'code.rtmpose3d': ['configs/*.py', 'LICENSE', 'UPSTREAM.md']},
))
