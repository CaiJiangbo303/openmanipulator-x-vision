#!/usr/bin/env python3
#
# Copyright 2024 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Wonho Yoon, Sungho Woo

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import ThisLaunchFileDir
from launch_ros.actions import Node


def is_valid_to_launch():
    # Path includes model name of Raspberry Pi series
    path = '/sys/firmware/devicetree/base/model'
    if os.path.exists(path):
        return False
    else:
        return True


def generate_launch_description():
    if not is_valid_to_launch():
        print('Can not launch fake robot in Raspberry Pi')
        return LaunchDescription([])

    start_rviz = LaunchConfiguration('start_rviz')
    prefix = LaunchConfiguration('prefix')
    use_fake_hardware = LaunchConfiguration('use_fake_hardware')
    fake_sensor_commands = LaunchConfiguration('fake_sensor_commands')
    startup_pose = LaunchConfiguration('startup_pose')
    joint1_delay_sec = LaunchConfiguration('joint1_delay_sec')
    joint1_duration_sec = LaunchConfiguration('joint1_duration_sec')
    remainder_duration_sec = LaunchConfiguration('remainder_duration_sec')

    return LaunchDescription([
        DeclareLaunchArgument(
            'start_rviz',
            default_value='true',
            description='Whether execute rviz2'),

        DeclareLaunchArgument(
            'prefix',
            default_value='""',
            description='Prefix of the joint and link names'),

        DeclareLaunchArgument(
            'use_fake_hardware',
            default_value='true',
            description='Start robot with fake hardware mirroring command to its states.'),

        DeclareLaunchArgument(
            'fake_sensor_commands',
            default_value='true',
            description='Enable fake command interfaces for sensors used for simple simulations. \
            Used only if "use_fake_hardware" parameter is true.'),

        DeclareLaunchArgument(
            'startup_pose', default_value='none', choices=['none', 'zero', 'home'],
            description='Software-only staged pose for RViz: zero, home, or none.'),
        DeclareLaunchArgument('joint1_delay_sec', default_value='1.0'),
        DeclareLaunchArgument('joint1_duration_sec', default_value='2.0'),
        DeclareLaunchArgument('remainder_duration_sec', default_value='4.0'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([ThisLaunchFileDir(), '/base.launch.py']),
            launch_arguments={
                'start_rviz': start_rviz,
                'prefix': prefix,
                # Fake hardware is a ros2_control test backend, not Gazebo.
                # Explicitly disable the base launch's default simulation mode
                # so that controller_manager is actually started.
                'use_sim': 'false',
                'use_fake_hardware': use_fake_hardware,
                'fake_sensor_commands': fake_sensor_commands,
            }.items(),
        ),
        Node(
            package='open_manipulator_x_bringup',
            executable='staged_pose_node.py',
            name='arm_pose_sequence',
            output='screen',
            parameters=[{
                'startup_pose': startup_pose,
                'joint1_delay_sec': joint1_delay_sec,
                'joint1_duration_sec': joint1_duration_sec,
                'remainder_duration_sec': remainder_duration_sec,
            }],
        ),
    ])
