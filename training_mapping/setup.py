from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'training_mapping'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        (os.path.join('share', package_name, 'launch'),
         glob(os.path.join('launch', '*.launch.py'))),

        (os.path.join('share', package_name, 'config'),
         glob(os.path.join('config', '*.yaml'))),

        (os.path.join('share', package_name, 'rviz'),
         glob(os.path.join('rviz', '*.rviz')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mbturton',
    maintainer_email='mbturton33@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 'gridmap_generator = training_mapping.gridmap_generator_node:main'
            'occupancy_map_generator = training_mapping.occupancy_map_generator_node:main'
        ],
    },
)
