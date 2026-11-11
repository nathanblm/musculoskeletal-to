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
            ['../../code/mmpose_pipeline.py', '../../code/arm_joint_state.py']),
        (f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/rtmpose3d',
            glob('../../code/rtmpose3d/*.py')
            + ['../../code/rtmpose3d/LICENSE', '../../code/rtmpose3d/UPSTREAM.md']),
        (f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/rtmpose3d/configs',
            glob('../../code/rtmpose3d/configs/*.py')),
    ],
    install_requires=['setuptools'],
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
            'pose_estimator = fatigue_control_ros.pose_estimator:main',
            'model_viewer = fatigue_control_ros.model_viewer:main',
        ],
    },
)
