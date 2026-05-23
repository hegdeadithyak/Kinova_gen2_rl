import os
from glob import glob
from setuptools import setup

package_name = 'point_goal_test'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Click a pixel in the RealSense depth feed, arm moves to that 3D point',
    license='MIT',
    entry_points={
        'console_scripts': [
            'point_selector_node = point_goal_test.point_selector_node:main',
            'arm_mover_node = point_goal_test.arm_mover_node:main',
        ],
    },
)
