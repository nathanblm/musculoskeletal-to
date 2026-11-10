from setuptools import find_packages, setup
from pathlib import Path
import os

package_name = 'human_pose'
model_dir = Path(__file__).resolve().parents[2] / 'musculoskeletal_models' / 'tsm_3DOF'


def source_path(path):
    return os.path.relpath(path, Path(__file__).resolve().parent)

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['README.md']),
        ('share/' + package_name + '/models/tsm_3DOF',
            [source_path(model_dir / 'tsm_3DOF.osim')]),
        ('share/' + package_name + '/models/tsm_3DOF/Geometry',
            [source_path(path) for path in sorted((model_dir / 'Geometry').glob('*.vtp'))]),
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
            'pose_publisher = human_pose.pose_publisher:main',
            'msk_model_visualizer = human_pose.msk_model_visualizer:main',
        ],
    },
)
