from setuptools import setup
import os
from glob import glob

package_name = 'tiago_social_cmd'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='daniel',
    maintainer_email='daniel@example.com',
    description='Package for executing social navigation commands',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'social_cmd_node = tiago_social_cmd.social_cmd_node:main'
        ],
    },
)
