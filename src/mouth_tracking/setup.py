from setuptools import find_packages, setup

package_name = 'mouth_tracking'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='amma',
    maintainer_email='hegdeadithyak@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'mouth_tracker = mouth_tracking.mouth_tracker:main',
            'room_mapper = mouth_tracking.room_mapper:main',
            'static_pointcloud = mouth_tracking.static_pointcloud:main'
        ],
    },
)
