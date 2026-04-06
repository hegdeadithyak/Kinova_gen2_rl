from setuptools import setup
import os
from glob import glob

setup(
    name='kinova_rl',
    version='0.1.0',
    packages=['kinova_rl'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/kinova_rl']),
        ('share/kinova_rl', ['package.xml']),
        (os.path.join('share', 'kinova_rl', 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    entry_points={
        'console_scripts': [
            'train    = kinova_rl.train:main',
            'enjoy    = kinova_rl.enjoy:main',
            'thoughts = kinova_rl.thoughts_node:main',
        ],
    },
)
