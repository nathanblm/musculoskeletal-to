from catkin_pkg.python_setup import generate_distutils_setup
from setuptools import setup

setup(**generate_distutils_setup(
    packages=[
        'human_musculoskeletal', 'human_musculoskeletal.code',
        'human_musculoskeletal.code.rtmpose3d',
        'human_musculoskeletal.rmr_solver',
    ],
    package_dir={
        'human_musculoskeletal': 'human_musculoskeletal',
    },
    package_data={
        'human_musculoskeletal.code.rtmpose3d':
            ['configs/*.py', 'LICENSE', 'UPSTREAM.md'],
    },
))
