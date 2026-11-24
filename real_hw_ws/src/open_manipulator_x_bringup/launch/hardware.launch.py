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

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import ThisLaunchFileDir
from launch_ros.actions import Node


def generate_launch_description():
    start_rviz = LaunchConfiguration('start_rviz')
    prefix = LaunchConfiguration('prefix')
    use_fake_hardware = LaunchConfiguration('use_fake_hardware')
    use_sim = LaunchConfiguration('use_sim')
    port_name = LaunchConfiguration('port_name')
    disable_torque_at_init = LaunchConfiguration('disable_torque_at_init')
    startup_pose = LaunchConfiguration('startup_pose')
    joint1_delay_sec = LaunchConfiguration('joint1_delay_sec')
    joint1_duration_sec = LaunchConfiguration('joint1_duration_sec')
    remainder_duration_sec = LaunchConfiguration('remainder_duration_sec')

    return LaunchDescription([
        DeclareLaunchArgument(
            'start_rviz',
            default_value='false',
            description='Whether execute rviz2'),

        DeclareLaunchArgument(
            'prefix',
            default_value='""',
            description='Prefix of the joint and link names'),

        DeclareLaunchArgument(
            'use_fake_hardware',
            default_value='false',
            description='Start robot with fake hardware mirroring command to its states.'),

        DeclareLaunchArgument(
            'use_sim',
            default_value='false',
            description='Start robot in Gazebo simulation.'),

        DeclareLaunchArgument(
            'port_name',
            default_value='/dev/ttyUSB0',
            description='The port name to connect to hardware.'),

        DeclareLaunchArgument(
            'disable_torque_at_init',
            default_value='false',
            choices=['true', 'false'],
            description=(
                'Disable already-enabled Dynamixel torque during initialization. '
                'Use only when the arm is physically supported.'),
        ),

        DeclareLaunchArgument(
            'startup_pose',
            default_value='none',
            choices=['none', 'zero', 'home'],
            description='Optional staged startup pose. Default none is passive.'),

        DeclareLaunchArgument(
            'joint1_delay_sec',
            default_value='1.0',
            description='Pause after joint1 reaches its target before joints 2-4 move.'),

        DeclareLaunchArgument(
            'joint1_duration_sec',
            default_value='2.0',
            description='Duration for the first joint1-only stage.'),

        DeclareLaunchArgument(
            'remainder_duration_sec',
            default_value='4.0',
            description='Duration for the remaining joints stage.'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([ThisLaunchFileDir(), '/base.launch.py']),
            launch_arguments={
                'start_rviz': start_rviz,
                'prefix': prefix,
                'use_fake_hardware': use_fake_hardware,
                'use_sim': use_sim,
                'port_name': port_name,
                'disable_torque_at_init': disable_torque_at_init,
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
