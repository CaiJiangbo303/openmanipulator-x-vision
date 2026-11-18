#!/usr/bin/env bash
cd ~/openmanipulator-x-vision/vision_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run grasp_command_bridge grasp_command_bridge \
  --ros-args \
  -p use_sim_time:=true \
  -p dry_run:=true

exec bash
