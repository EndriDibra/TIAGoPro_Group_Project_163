from setuptools import setup
import os
from glob import glob

package_name = 'human_spawner'

data_files = [
    ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
]

# Recursively add models
for dirpath, dirnames, filenames in os.walk('models'):
    if filenames:
        install_path = os.path.join('share', package_name, dirpath)
        data_files.append((install_path, [os.path.join(dirpath, f) for f in filenames]))
setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='ROS2 package for spawning and moving human models in Gazebo',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'human_controller = human_spawner.human_controller:main',
        ],
    },
)
