from pathlib import Path

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

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

    state_manager = Node(
        package='object_state_manager',
        executable='object_state_manager',
        name='object_state_manager',
        output='screen',
        parameters=[
            {
                'use_sim_time': False,

                'world_frame': frame,

                'fast_measurement_topic':
                    '/object_localizer/detections_3d_fast',

                'precise_measurement_topic':
                    '/object_localizer/detections_3d_mask',

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

    return LaunchDescription([
        fast3d,
        mask3d,
        state_manager,
    ])
