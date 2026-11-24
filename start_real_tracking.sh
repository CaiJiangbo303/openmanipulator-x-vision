#!/usr/bin/env bash
set -e

source /opt/ros/humble/setup.bash
source ~/openmanipulator-x-vision/real_hw_ws/install/setup.bash
source ~/openmanipulator-x-vision/vision_ws/install/setup.bash

export ROS_DOMAIN_ID=189
export PYTHONPATH="$HOME/manipulator-x/grasp_ws/src/manipulator_grasp:$PYTHONPATH"

cd ~/openmanipulator-x-vision

echo "===== STOP OLD TRACKER ====="
pkill -INT -f real_xy_tracker_stage1.py 2>/dev/null || true
pkill -INT -f real_visual_tracker_final.py 2>/dev/null || true
sleep 1

echo "===== ARM ====="
if ! ros2 action list | grep -qx '/arm_controller/follow_joint_trajectory'; then
    echo "ARM ACTION SERVER = FAIL"
    exit 1
fi
echo "ARM ACTION SERVER = PASS"

echo "===== CAMERA ====="
if ! timeout 3 ros2 topic echo --once \
    /camera/camera/color/image_raw \
    >/dev/null 2>&1
then
    echo "CAMERA STREAM = FAIL"
    echo "Restart RealSense first."
    exit 1
fi
echo "CAMERA STREAM = PASS"

echo "===== MASK3D ====="
if ! timeout 4 ros2 topic echo --once \
    /object_localizer/detections_3d_mask \
    >/dev/null 2>&1
then
    echo "MASK3D = FAIL"
    echo "Perception pipeline is not producing data."
    exit 1
fi
echo "MASK3D = PASS"

echo "===== CAMERA WORLD TF ====="

if ! ros2 node list \
    | grep -qx '/camera_world_static_tf'
then
    echo "Starting calibrated camera TF..."

    python3 \
      run_nodes/publish_calibrated_camera_tf.py \
      >/tmp/camera_world_static_tf.log 2>&1 &

    sleep 2
fi

if ! timeout 3 ros2 run tf2_ros tf2_echo \
    world camera_color_optical_frame \
    2>/dev/null \
    | grep -q 'Translation'
then
    echo "CAMERA WORLD TF = FAIL"
    exit 1
fi

echo "CAMERA WORLD TF = PASS"

echo
echo "=========================================="
echo " FAST CONTINUOUS TRACKING START"
echo " Ctrl+C = STOP"
echo " Power Hub = EMERGENCY STOP"
echo "=========================================="
echo

exec python3 \
  run_nodes/real_visual_tracker_final.py
