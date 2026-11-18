#!/usr/bin/env python3

from pathlib import Path

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node


def find_latest_model():
    validation_dir = Path.home() / 'manipulator-x-runtime' / 'validation'

    candidates = list(
        validation_dir.glob('session_*/model/model.sdf')
    )

    if not candidates:
        raise RuntimeError(
            'No generated F550 model found under '
            '~/manipulator-x-runtime/validation/session_*/model/'
        )

    model_sdf = max(
        candidates,
        key=lambda p: p.stat().st_mtime
    )

    robot_urdf = model_sdf.parent / 'robot.urdf'

    if not robot_urdf.exists():
        raise RuntimeError(
            f'robot.urdf not found beside {model_sdf}'
        )

    return model_sdf, robot_urdf


def generate_launch_description():

    vision_sim_share = Path(
        get_package_share_directory('vision_sim')
    )

    world_file = (
        vision_sim_share /
        'worlds' /
        'vision_test.sdf'
    )

    model_sdf, robot_urdf = find_latest_model()

    robot_description = robot_urdf.read_text()

    # --------------------------------------------------
    # 1. Gazebo Fortress server
    # --------------------------------------------------

    gazebo = ExecuteProcess(
        cmd=[
            'ign',
            'gazebo',
            '-r',
            '-s',
            '-v', '2',
            '--render-engine', 'ogre',
            str(world_file),
        ],
        output='screen'
    )

    # --------------------------------------------------
    # 2. Robot State Publisher
    # fixed TF:
    # base_link -> camera -> optical frame
    # --------------------------------------------------

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='vision_robot_state_publisher',
        output='screen',
        parameters=[
            {
                'robot_description': robot_description,
                'use_sim_time': True,
            }
        ]
    )

    # --------------------------------------------------
    # 3. Gazebo camera bridge
    # --------------------------------------------------

    camera_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='vision_camera_bridge',
        output='screen',
        arguments=[
            '/camera/image@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/camera/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo',
        ]
    )

    # --------------------------------------------------
    # 4. Gazebo clock + model pose bridge
    # --------------------------------------------------

    pose_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='vision_pose_bridge',
        output='screen',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
            '/world/manipulator_x/pose/info@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V',
            '--ros-args',
            '-r',
            '/world/manipulator_x/pose/info:=/simulation/poses',
        ]
    )

    # --------------------------------------------------
    # 5. world -> F550 dynamic TF
    # --------------------------------------------------

    simulation_tf = Node(
        package='vision_sim',
        executable='simulation_tf.py',
        name='vision_sim_localization',
        output='screen'
    )

    # --------------------------------------------------
    # 6. Spawn F550 + inverted OpenMANIPULATOR-X
    #
    # Current validated pose:
    # x=-0.25, y=0, z=0.80
    # --------------------------------------------------

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        name='spawn_f550',
        output='screen',
        arguments=[
            '-world', 'manipulator_x',
            '-file', str(model_sdf),
            '-name', 'f550_high',
            '-x', '-0.25',
            '-y', '0.0',
            '-z', '0.80',
        ]
    )

    # --------------------------------------------------
    # 7. Our RGB-D frontend
    # --------------------------------------------------

    rgbd_frontend = Node(
        package='rgbd_frontend',
        executable='rgbd_frontend',
        name='rgbd_frontend',
        output='screen',
        parameters=[
            {
                'use_sim_time': True,
            }
        ]
    )


    simulation_arm_tf_bridge = Node(
        package='vision_sim',
        executable='simulation_arm_tf.py',
        name='simulation_arm_tf_bridge',
        output='screen',
        parameters=[
            {
                'use_sim_time': True
            }
        ],
    )

    return LaunchDescription([
        simulation_arm_tf_bridge,

        gazebo,

        robot_state_publisher,

        pose_bridge,

        simulation_tf,

        # Give Gazebo time to create the world first.
        TimerAction(
            period=3.0,
            actions=[
                spawn_robot,
            ]
        ),

        # Camera bridge can start shortly after spawn.
        TimerAction(
            period=4.0,
            actions=[
                camera_bridge,
            ]
        ),

        # Start perception after camera topics exist.
        TimerAction(
            period=5.0,
            actions=[
                rgbd_frontend,
            ]
        ),
    ])
