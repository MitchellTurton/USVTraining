from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node

import os


def generate_launch_description() -> LaunchDescription:

    pkg_share = get_package_share_directory('training_mapping')
    localization = get_package_share_directory('training_localization')
    perception = get_package_share_directory('training_perception')

    usv_arg = DeclareLaunchArgument('usv', default_value='simulation')
    usv_config = LaunchConfiguration('usv', default='simulation')

    occupancy_map_generator = (pkg_share, '/config/gridmap.yaml')

    return LaunchDescription([
        usv_arg,

        # Localization Launch
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(localization, 'launch', 'ekf.launch.py')
            ),
            launch_arguments={'usv': usv_config}.items()
        ),

        # Perception Launch
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(perception, 'launch', 'main.launch.py')
            ),
            launch_arguments={'usv': usv_config}.items()
        ),

        # Creating the GridGenerator Node
        Node(
            package='training_mapping',
            executable='occupancy_map_generator',
            parameters=[occupancy_map_generator]
        ),

        # Rviz launcher
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_share, 'launch', 'rviz.launch.py')
            )
        )
    ])
