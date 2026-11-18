#!/usr/bin/env bash
cd ~/openmanipulator-x-vision/vision_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

clear

echo "======================================"
echo " OpenManipulator-X Monitor"
echo "======================================"
echo
echo "System is running."
echo
echo "Useful commands:"
echo
echo "ros2 node list | sort"
echo
echo "ros2 topic echo /object_localizer/detections_3d_mask --once"
echo
echo "ros2 topic echo /target_selector/selection --once"
echo
echo "ros2 topic echo /grasp_command_bridge/status --once"
echo
echo "pgrep -af 'ign gazebo'"
echo

exec bash
