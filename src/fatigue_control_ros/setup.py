from glob import glob
import sys

from setuptools import find_packages, setup

package_name = 'fatigue_control_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',
            glob('launch/*.launch.py') + glob('launch/*.launch.xml')),
        (f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages',
            ['../../code/mmpose_pipeline.py']),
        (f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/code',
            glob('../../code/*.py')),
        (f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/rmr_solver',
            glob('../../rmr_solver/*.py')),
        (f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/code/visuals',
            glob('../../code/visuals/*.py')),
        (f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/visuals',
            ['../../code/visuals/__init__.py',
             '../../code/visuals/live_joint_velocity.py',
             '../../code/visuals/live_muscle_fatigue.py']),
        (f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/rtmpose3d',
            glob('../../code/rtmpose3d/*.py')
            + ['../../code/rtmpose3d/LICENSE', '../../code/rtmpose3d/UPSTREAM.md']),
        (f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/rtmpose3d/configs',
            glob('../../code/rtmpose3d/configs/*.py')),
    ],
    install_requires=['setuptools', 'matplotlib'],
    zip_safe=True,
    maintainer='nathan',
    maintainer_email='n.blum@student.tudelft.nl',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'human_state_estimator = fatigue_control_ros.human_state_estimator:main',
            'model_viewer = fatigue_control_ros.model_viewer:main',
        ],
    },
)
