from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def build_nodes(context):

    root = Path(__file__).resolve().parents[1]

    config = str(
        root
        / 'vision_ws/src/object_localizer/config/realsense.yaml'
    )

    # Perception-only bench mode.
    # Both measurement sources must use the same frame.
    frame = 'camera_color_optical_frame'

    fast3d = Node(
        package='object_localizer',
        executable='object_localizer',
        name='object_localizer',
        output='screen',
        parameters=[
            config,
            {
                'use_sim_time': False,
                'world_frame': frame,
                'output_topic':
                    '/object_localizer/detections_3d_fast',
            }
        ]
    )

    mask3d = Node(
        package='object_localizer',
        executable='mask_localizer',
        name='mask_localizer',
        output='screen',
        parameters=[
            config,
            {
                'use_sim_time': False,
                'world_frame': frame,
                'output_topic':
                    '/object_localizer/detections_3d_mask',
            }
        ]
    )

    enable_filter = IfCondition(
        LaunchConfiguration('enable_self_filter')
    ).evaluate(context)

    fast_topic = '/object_localizer/detections_3d_fast'
    precise_topic = '/object_localizer/detections_3d_mask'

    filters = []

    if enable_filter:

        def make_filter(name, input_topic, output_topic, timeout):

            return Node(
                package='object_self_filter',
                executable='object_self_filter',
                name=name,
                output='screen',
                parameters=[{
                    'use_sim_time': False,
                    'world_frame': frame,
                    'input_topic': input_topic,
                    'output_topic': output_topic,
                    'tf_wait_timeout': timeout,
                }]
            )

        filters = [
            make_filter(
                'self_filter_fast',
                fast_topic,
                '/object_self_filter/detections_3d_fast',
                0.12
            ),
            make_filter(
                'self_filter_precise',
                precise_topic,
                '/object_self_filter/detections_3d_mask',
                0.20
            ),
        ]

        fast_topic = '/object_self_filter/detections_3d_fast'
        precise_topic = '/object_self_filter/detections_3d_mask'

    state_manager = Node(
        package='object_state_manager',
        executable='object_state_manager',
        name='object_state_manager',
        output='screen',
        parameters=[
            {
                'use_sim_time': False,

                'world_frame': frame,

                'fast_measurement_topic': fast_topic,

                'precise_measurement_topic': precise_topic,

                'output_topic':
                    '/object_state_manager/states',

                'publish_rate_hz': 20.0,

                'fast_history_size': 32,

                'precise_match_tolerance_ms': 20.0,

                'precise_bias_alpha': 0.50,

                'precise_bias_timeout': 1.50,

                'stale_timeout': 0.50,

                'lost_timeout': 1.50,

                'remove_timeout': 3.0,

                'max_prediction_time': 1.00,
            }
        ]
    )

    state_gate = Node(
        package='target_selector',
        executable='state_gate',
        name='state_gate',
        output='screen',
        condition=IfCondition(
            LaunchConfiguration('enable_state_gate')
        ),
        parameters=[
            {
                'use_sim_time': False,
                'frame': frame,
                'input_topic':
                    '/object_state_manager/states',
                'status_topic':
                    '/state_gate/status',
                'requested_track_id':
                    LaunchConfiguration('requested_track_id'),
                'allowed_classes':
                    'blue block,cell phone,smartphone,mobile phone',
                'min_confidence': 0.50,
                'max_uncertainty': 0.030,
                'max_age_ms': 350.0,
                'max_prediction_age_ms': 250.0,
                'watchdog_ms': 250.0,
            }
        ]
    )

    return [
        state_gate,
        fast3d,
        mask3d,
        *filters,
        state_manager,
    ]


def generate_launch_description():

    return LaunchDescription([
        DeclareLaunchArgument(
            'enable_state_gate',
            default_value='false'
        ),
        DeclareLaunchArgument(
            'requested_track_id',
            default_value=''
        ),
        DeclareLaunchArgument(
            'enable_self_filter',
            default_value='false'
        ),
        OpaqueFunction(function=build_nodes),
    ])
